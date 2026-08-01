#!/usr/bin/env -S uv run --with huggingface_hub python
"""Pick best-rank LoRA per genre from the rank sweep, stage the weights, write
an updated model card (with full sweep table + dataset note + selection
rationale), and upload to HuggingFace.

Reads:
  ai/results/lora_rank_sweep.csv   (genre,rank,val_loss,val_top1,...)
  ai/results/f1_per_genre_baseline.csv

Writes:
  model/ft_f1_lora_<genre>/*       (staged adapter + tokenizer + code + README.md)

Uploads:
  PearlLeeStudio/TheArtist-MusicTransformer-lora-<genre>   (overwrite=False on
                                                            existing files)

Selection criterion: lowest validation cross-entropy loss; val_top1 tiebreak.
val_loss matches the training-time best.pt selection so this stays consistent
with how each individual checkpoint was chosen.

Flags:
  --dry-run     plan + would-stage logs only, no disk writes, no upload
  --plan-only   print best-rank plan and exit (no staging, no upload)
  --no-upload   stage + write card, but skip HF push
  --only g1,g2  subset of genres
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = REPO_DIR / "ai" / "checkpoints"
SWEEP_CSV = REPO_DIR / "ai" / "results" / "lora_rank_sweep.csv"
BASELINE_CSV = REPO_DIR / "ai" / "results" / "f1_per_genre_baseline.csv"
REAL_SONG_CSV = REPO_DIR / "ai" / "results" / "real_song_eval.csv"
TRAIN_CODE_DIR = REPO_DIR / "ai" / "training"

ORG = "PearlLeeStudio"
LORA_GENRES = [
    "country", "funk", "gospel", "rnb_soul", "hip_hop",
    "electronic", "folk", "classical", "rock", "blues", "bossa",
]
BASE_REPO = f"{ORG}/TheArtist-MusicTransformer-ft-pop80"

# Same as _generate_lora_cards.py — single source of truth for descriptive
# fields. (Kept inline to avoid an awkward sibling-script import.)
GENRE_INFO: dict[str, tuple[str, int, str]] = {
    "country":    ("Country",     53_000, "Country pop, country rock, and modern country idioms"),
    "funk":       ("Funk",         2_837, "Funk grooves with extended dominants and altered sevenths"),
    "gospel":     ("Gospel",       3_747, "Gospel cadences, plagal motion, and gospel substitutions"),
    "rnb_soul":   ("R&B / Soul",   9_550, "R&B and soul harmony — subdominant minor, suspensions, modal interchange"),
    "hip_hop":    ("Hip-Hop",     14_021, "Hip-hop chord conventions — minor-key, neo-soul-adjacent harmony"),
    "electronic": ("Electronic",  15_196, "Electronic harmony — minor-key, repetitive vamps, tonic-centred motion"),
    "folk":       ("Folk",        60_752, "Folk and singer-songwriter harmony"),
    "classical":  ("Classical",    1_000, "Bach chorales (curated subset from the music21 corpus)"),
    "rock":       ("Rock",        48_910, "Classic rock and modern rock harmony"),
    "blues":      ("Blues",        9_944, "12-bar and modified blues forms"),
    "bossa":      ("Bossa nova",  14_315, "Brazilian bossa nova jazz harmony"),
}

# Exact splits/train.jsonl row counts as of 2026-05-15 (the LoRA training data
# every model in this release saw). Sourced from
# `python3 -c "import json; from collections import Counter; ..."` on splits.
GENRE_SPLITS: dict[str, dict[str, int]] = {
    "country":    {"train": 49_388, "val": 6_173, "test": 6_175},
    "folk":       {"train": 48_601, "val": 6_075, "test": 6_076},
    "rock":       {"train": 39_128, "val": 4_891, "test": 4_891},
    "electronic": {"train": 12_156, "val": 1_519, "test": 1_521},
    "bossa":      {"train": 11_452, "val": 1_431, "test": 1_432},
    "hip_hop":    {"train": 11_216, "val": 1_402, "test": 1_403},
    "blues":      {"train":  7_955, "val":   994, "test":   995},
    "rnb_soul":   {"train":  7_640, "val":   955, "test":   955},
    "gospel":     {"train":  2_997, "val":   374, "test":   376},
    "funk":       {"train":  2_269, "val":   283, "test":   285},
    "classical":  {"train":    296, "val":    37, "test":    38},
}

# Exact filter rules in ai/training/extract_genre_subsets.py:GENRE_FILTERS.
# `main` matches `main_genre` column; `genres_any` substring-matches the
# `genres` column (Chordonomicon's free-form list). first-match-only across
# genres so a song never double-counts.
GENRE_FILTER_DESC: dict[str, str] = {
    "country":    "`main_genre` ∈ {country} OR `genres` contains any of {country pop, country rock, modern country}",
    "funk":       "`genres` contains any of {funk}",
    "gospel":     "`genres` contains any of {gospel}",
    "rnb_soul":   "`main_genre` ∈ {soul} OR `genres` contains any of {r&b, rnb, soul, motown, neo soul}",
    "hip_hop":    "`main_genre` ∈ {rap} OR `genres` contains any of {hip hop, hip-hop, trap, rap}",
    "electronic": "`main_genre` ∈ {electronic} OR `genres` contains any of {house, techno, edm, trance, synthwave, dance, electro, drum and bass, dnb}",
    "folk":       "`genres` contains any of {folk, singer-songwriter, acoustic}",
    "rock":       "`main_genre` ∈ {rock}",
    "blues":      "`genres` contains any of {blues}",
    "bossa":      "`genres` contains any of {bossa, samba, latin, salsa, cumbia}",
    "classical":  "Bach chorales from `music21.corpus` — Chordonomicon filter does NOT apply (no classical genre in Chordonomicon)",
}

# Per-genre 130-song eval-set composition. Populated from
# ai/data/eval_real_songs.jsonl at runtime (so re-running the build script
# with a different seed/per-genre count is automatically reflected).

DATA_NOTE_CHORDONOMICON = (
    "Chordonomicon is licensed CC BY-NC 4.0; see the dataset card for full terms."
)
DATA_NOTE_CLASSICAL = (
    "Bach chorales come from the public-domain music21 corpus and are"
    " license-clean for commercial use; the CC BY-NC restriction below"
    " applies only because the LoRA's runtime is loaded on top of an"
    " F1 base that itself trained on Chordonomicon."
)

# d_model=512, n_layers=8, target_modules={w_q,w_k,w_v} → each LoRA layer has
# A (d_in×r) + B (r×d_out) parameters. The trained embedding extension adds
# 8 (extra genre tokens) × 512 × 2 ≈ 8K independent of rank.
N_LAYERS = 8
D_MODEL = 512
N_TARGETS = 3  # w_q, w_k, w_v
EMB_EXT_PARAMS = 8 * D_MODEL * 2  # ≈ 8,192 (token_emb + out_proj rows)


def lora_trainable_params(rank: int) -> int:
    """LoRA-only trainable params + embedding extension. Matches the
    `Trainable params:` line emitted by lora_train.py."""
    lora = N_LAYERS * N_TARGETS * (rank * D_MODEL + rank * D_MODEL)  # A + B per target per layer
    return lora + EMB_EXT_PARAMS


def trainable_pct(rank: int) -> float:
    """Trainable / total post-resize. Base ~25.85M; resized vocab adds tiny."""
    return 100.0 * lora_trainable_params(rank) / 26_000_000


def load_sweep() -> dict[str, list[dict]]:
    """Returns {genre: [{rank, val_loss, val_ppl, val_top1, val_top5, val_sequences}, ...]}.
    Skips `rank == 'full_ft'` rows — those are anchors, not part of the LoRA
    rank selection."""
    if not SWEEP_CSV.exists():
        sys.exit(f"[fail] sweep CSV missing: {SWEEP_CSV}")
    by_genre: dict[str, list[dict]] = {}
    with open(SWEEP_CSV) as f:
        for r in csv.DictReader(f):
            if r["rank"] == "full_ft":
                continue
            by_genre.setdefault(r["genre"], []).append({
                "rank": int(r["rank"]),
                "val_loss": float(r["val_loss"]),
                "val_ppl": float(r["val_ppl"]),
                "val_top1": float(r["val_top1"]),
                "val_top5": float(r["val_top5"]),
                "val_sequences": int(r["val_sequences"]),
            })
    for g in by_genre:
        by_genre[g].sort(key=lambda r: r["rank"])
    return by_genre


def load_baseline() -> dict[str, dict]:
    if not BASELINE_CSV.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(BASELINE_CSV) as f:
        for r in csv.DictReader(f):
            rows[r["genre"]] = {
                "val_loss": float(r["val_loss"]),
                "val_ppl": float(r["val_ppl"]),
                "val_top1": float(r["val_top1"]),
                "val_top5": float(r["val_top5"]),
                "val_sequences": int(r["val_sequences"]),
            }
    return rows


def load_real_song_eval_set_per_genre() -> dict[str, dict]:
    """Read the 130-song eval set JSONL and return per-genre composition stats:
    {genre: {n, sources, named, bar_min, bar_max, avg_dur_sec}}. Used to fill
    in the 'Curated 130-song eval' subsection of each card."""
    import json
    out: dict[str, dict] = {}
    p = REPO_DIR / "ai" / "data" / "eval_real_songs.jsonl"
    if not p.exists():
        return out
    by_genre: dict[str, list[dict]] = {}
    for line in open(p):
        d = json.loads(line)
        by_genre.setdefault(d["genre"], []).append(d)
    for g, rows in by_genre.items():
        bars = sorted(r["n_bars"] for r in rows)
        out[g] = {
            "n": len(rows),
            "sources": sorted({r["source"] for r in rows}),
            "named": sum(1 for r in rows if r.get("title_is_real")),
            "bar_min": bars[0],
            "bar_max": bars[-1],
            "avg_dur_sec": round(sum(r["est_duration_sec"] for r in rows) / len(rows)),
        }
    return out


def load_real_song_eval() -> dict[tuple[str, str], dict]:
    """Aggregates ai/results/real_song_eval.csv to {(model, genre) → mean metrics}.
    Returns empty dict if the CSV hasn't been generated yet (eval_real_songs.py
    hasn't run yet). Falling back gracefully keeps the release script usable
    standalone."""
    import statistics
    if not REAL_SONG_CSV.exists():
        return {}
    bucket: dict[tuple[str, str], list[dict]] = {}
    with open(REAL_SONG_CSV) as f:
        for r in csv.DictReader(f):
            try:
                top1 = float(r["val_top1"])
                top5 = float(r["val_top5"])
                loss = float(r["val_loss"])
            except (ValueError, KeyError):
                continue
            if top1 != top1:  # NaN
                continue
            key = (r["model"], r["genre"])
            bucket.setdefault(key, []).append({"top1": top1, "top5": top5, "loss": loss})
    out: dict[tuple[str, str], dict] = {}
    for key, lst in bucket.items():
        out[key] = {
            "n": len(lst),
            "top1": statistics.mean(l["top1"] for l in lst),
            "top5": statistics.mean(l["top5"] for l in lst),
            "loss": statistics.mean(l["loss"] for l in lst),
        }
    return out


def render_real_song_section(genre: str, real_song_eval: dict[tuple[str, str], dict]) -> str:
    """Render the 'Real-song eval' markdown section for this LoRA. Returns
    empty string when no eval data is available — the section is omitted
    entirely until eval_real_songs.py has run."""
    if not real_song_eval:
        return ""
    lora_model = f"LoRA-{genre}"
    own = real_song_eval.get((lora_model, genre))
    f1 = real_song_eval.get(("F1", genre))
    if own is None and f1 is None:
        return ""
    head = (
        "## Real-song eval\n\n"
        f"Mean validation top-1/top-5/cross-entropy on **{own['n'] if own else f1['n']} held-out real {genre} songs** "
        f"from `ai/data/eval_real_songs.jsonl` (held-out from `ai/data/splits/{{val,test}}.jsonl`, "
        "see the released model cards "
        "for dataset composition + methodology). Teacher-forced eval — same `evaluate()` call as the full-val "
        "rank-sweep eval above, just narrowed to a curated 10-song subset.\n\n"
    )
    tbl = ["| Model | Top-1 (%) | Top-5 (%) | val_loss |", "|---|---:|---:|---:|"]
    if f1 is not None:
        tbl.append(f"| F1 base alone | {f1['top1']:.2f} | {f1['top5']:.2f} | {f1['loss']:.4f} |")
    if own is not None:
        tbl.append(f"| **F1 + this LoRA** | **{own['top1']:.2f}** | {own['top5']:.2f} | {own['loss']:.4f} |")
    if own is not None and f1 is not None:
        d1 = own["top1"] - f1["top1"]
        d5 = own["top5"] - f1["top5"]
        dl = own["loss"] - f1["loss"]
        tbl.append(f"| Δ | {d1:+.2f} | {d5:+.2f} | {dl:+.4f} |")
    return head + "\n".join(tbl) + "\n\n"


def pick_best(rows: list[dict]) -> dict:
    """Min val_loss; val_top1 tiebreak (higher wins)."""
    return min(rows, key=lambda r: (r["val_loss"], -r["val_top1"]))


def _src_dir_for(genre: str, rank: int) -> Path:
    suffix = "" if rank == 8 else f"_r{rank}"
    return CHECKPOINTS_DIR / f"ft_f1_lora_{genre}{suffix}"


def stage_weights(genre: str, rank: int, dry_run: bool) -> Path:
    src = _src_dir_for(genre, rank)
    dst = MODEL_DIR / f"ft_f1_lora_{genre}"

    if not (src / "adapter" / "adapter_config.json").exists():
        sys.exit(f"[fail] missing adapter at {src}/adapter — checkpoint incomplete")

    weight_files: list[tuple[Path, str]] = [
        (src / "adapter" / "adapter_config.json",        "adapter_config.json"),
        (src / "adapter" / "adapter_model.safetensors",  "adapter_model.safetensors"),
        (src / "config.json",                             "config.json"),
        (src / "tokenizer.json",                          "tokenizer.json"),
        (src / "embedding_extension.pt",                  "embedding_extension.pt"),
    ]
    code_files: list[tuple[Path, str]] = [
        (TRAIN_CODE_DIR / "model.py",     "model.py"),
        (TRAIN_CODE_DIR / "tokenizer.py", "tokenizer.py"),
    ]

    print(f"  stage   {src.name} → model/{dst.name}/  ({len(weight_files) + len(code_files)} files)")
    if dry_run:
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for src_path, name in weight_files + code_files:
        shutil.copy2(src_path, dst / name)
    return dst


def adapter_size_mb(genre: str) -> float:
    p = MODEL_DIR / f"ft_f1_lora_{genre}" / "adapter_model.safetensors"
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def _format_rank_row(rank: int, row: dict | None, baseline_top1: float | None, selected: bool) -> str:
    label = f"r={rank}"
    if selected:
        label = f"**r={rank}**"
    if row is None:
        return f"| {label} | — | — | — | — |"
    delta = ""
    if baseline_top1 is not None:
        d = row["val_top1"] - baseline_top1
        sign = "+" if d >= 0 else ""
        delta = f"{sign}{d:.2f}"
    selected_mark = " ← selected" if selected else ""
    return (
        f"| {label} | {row['val_loss']:.4f} | {row['val_top1']:.2f} | "
        f"{row['val_top5']:.2f} | {delta}{selected_mark} |"
    )


def render_card(genre: str, best: dict, all_ranks: list[dict], baseline: dict,
                real_song_eval: dict[tuple[str, str], dict] | None = None,
                eval_set_stats: dict[str, dict] | None = None) -> str:
    info = GENRE_INFO[genre]
    display_name, count, blurb = info
    rank = best["rank"]
    alpha = rank * 2  # all configs in this project use alpha = 2 × rank
    epochs = 8 if genre in {"funk", "gospel", "blues", "classical"} else 5
    repo_id = f"{ORG}/TheArtist-MusicTransformer-lora-{genre.replace('_', '-')}"
    data_note = DATA_NOTE_CLASSICAL if genre == "classical" else DATA_NOTE_CHORDONOMICON

    n_trainable = lora_trainable_params(rank)
    pct_trainable = trainable_pct(rank)
    adapter_mb = adapter_size_mb(genre)
    real_song_section = render_real_song_section(genre, real_song_eval or {})

    # Training data + eval data details
    splits = GENRE_SPLITS.get(genre, {"train": 0, "val": 0, "test": 0})
    filter_desc = GENRE_FILTER_DESC.get(genre, "(filter rule not documented)")
    n_train, n_val, n_test = splits["train"], splits["val"], splits["test"]
    n_train_aug = n_train * 12  # 12-key transpose
    rank_suffix = "" if rank == 8 else f"_r{rank}"

    es = (eval_set_stats or {}).get(genre, {})
    eval_set_n = es.get("n", "—")
    eval_set_sources = ", ".join(f"`{s}`" for s in es.get("sources", [])) if es else "—"
    eval_set_named = es.get("named", "—")
    eval_set_bar_min = es.get("bar_min", "—")
    eval_set_bar_max = es.get("bar_max", "—")
    eval_set_avg_dur = es.get("avg_dur_sec", "—")

    # Rank sweep table — one row per rank in {4,8,16,32,64}
    rank_index = {r["rank"]: r for r in all_ranks}
    baseline_top1 = baseline.get("val_top1") if baseline else None
    sweep_rows: list[str] = []
    for r in (4, 8, 16, 32, 64):
        sweep_rows.append(_format_rank_row(r, rank_index.get(r), baseline_top1, selected=(r == rank)))
    sweep_table = "\n".join(sweep_rows)

    # Eval delta vs F1 baseline (no LoRA)
    f1_top1 = baseline.get("val_top1", 0.0) if baseline else 0.0
    f1_top5 = baseline.get("val_top5", 0.0) if baseline else 0.0
    f1_loss = baseline.get("val_loss", 0.0) if baseline else 0.0
    delta_top1 = best["val_top1"] - f1_top1
    delta_top5 = best["val_top5"] - f1_top5
    delta_loss = best["val_loss"] - f1_loss

    return f"""---
license: cc-by-nc-4.0
library_name: peft
pipeline_tag: text-generation
base_model: {BASE_REPO}
tags:
  - music
  - music-generation
  - chord-generation
  - symbolic-music
  - music-transformer
  - lora
  - peft
  - {genre.replace('_', '-')}
language:
  - en
---

# TheArtist Music Transformer — LoRA Adapter ({display_name})

**LoRA adapter that conditions the F1 base ({BASE_REPO}) toward {display_name.lower()} chord progressions.** One of eleven per-genre adapters released alongside the paper *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026). This release is the **best-rank** snapshot from a 5-point rank sweep (`r ∈ {{4, 8, 16, 32, 64}}`); see §Rank sweep below for the full table and selection criterion.

## Demo

[Watch TheArtist in action on YouTube](https://youtu.be/iSDj4VK9nKc) — interactive staff editor, MIDI input, AI generation with live progress, and per-genre LoRA playback across the 13-genre vocabulary.

## Adapter summary

| Field | Value |
|---|---|
| Base model | `{BASE_REPO}` (F1, 25.6M params) |
| Adapter type | LoRA (Q/K/V projections) |
| LoRA rank | **{rank}** |
| LoRA alpha | {alpha} |
| LoRA dropout | 0.05 |
| Target modules | `w_q`, `w_k`, `w_v` |
| Trainable parameters | ~{n_trainable:,} (~{pct_trainable:.2f}% of base) |
| Adapter file size | ~{adapter_mb:.1f} MB |
| Base vocabulary | 351 tokens (jazz/pop) |
| Vocabulary extension | +8 genre tokens (`embedding_extension.pt`) |
| Training epochs | {epochs} |

## Training data

### Source

{count:,} chord-progression sequences in the **{display_name.lower()}** subset of the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon). {data_note}

### Filter rule

{filter_desc}

(See `ai/training/extract_genre_subsets.py:GENRE_FILTERS` for the full extraction logic — `main` matches the `main_genre` column, `genres_any` substring-matches the free-form `genres` column. Each song is assigned to its first matching genre so it never double-counts.)

### Splits (song-level, seed=42, 80/10/10)

| Partition | Songs | Used for |
|---|---:|---|
| train | {n_train:,} | this LoRA's training (12-key augmented → {n_train_aug:,} sequences) |
| val | {n_val:,} | rank-sweep eval + best-epoch selection during training |
| test | {n_test:,} | held aside for future paired analysis |

### Vocabulary

- **Base**: 351 tokens (jazz/pop chord vocab from the F1 base model)
- **Extension**: +8 `[GENRE:X]` tokens covering 8 new genres (this LoRA adds the `[GENRE:{genre}]` token)
- **Final vocab**: 359 tokens (stored alongside the adapter in `embedding_extension.pt`)

### Reproducibility

```bash
# 1. Pull Chordonomicon raw csv into ai/data/raw/chordonomicon/
# 2. Extract this genre subset
uv run python ai/training/extract_genre_subsets.py --genres {genre} --merge

# 3. Train the LoRA at the released rank
uv run python ai/training/lora_train.py --config ai/training/configs/lora/{genre}{rank_suffix}.yaml
```

Hyperparameters: {epochs} epochs · batch 32 × accum 2 · lr 3e-4 · 1-epoch warmup · AMP fp16 · best.pt selected by min val_loss.

## Genre character

{blurb}

## Rank sweep

The released adapter is the **best-rank** snapshot from training the same LoRA recipe at five different ranks. Every cell uses the same F1 base, same val split, same `evaluate()` call, and the same `[GENRE:none]`-initialized embedding extension — only `lora_r` (and `lora_alpha = 2 × lora_r`) changes. Numbers are validation-set token-level metrics (no key augmentation).

| Rank | val_loss | val_top1 (%) | val_top5 (%) | Δtop1 vs F1 |
|---|---:|---:|---:|---:|
{sweep_table}

**Selection criterion**: minimum validation cross-entropy loss; `val_top1` as tiebreaker. `val_loss` is what the training loop optimizes and what selects each rank's `best.pt` epoch, so using it for cross-rank selection keeps consistency with how each individual checkpoint was chosen.

Full 11-genre × 5-rank sweep + full-FT anchor table: `ai/results/lora_rank_sweep.md` in the repo.

## Evaluation

Validation token-level metrics on the genre-specific val split ({best["val_sequences"]} sequences, no key augmentation). The F1 base column uses the same val split, same dataloader, and the same `[GENRE:none]`-initialized embedding-extension setup as the LoRA run — only the LoRA parameters and the trained embedding rows differ.

| Metric | F1 base alone | F1 + this LoRA | Δ |
|---|---:|---:|---:|
| Top-1 accuracy (%) | {f1_top1:.2f} | **{best["val_top1"]:.2f}** | **{delta_top1:+.2f}** |
| Top-5 accuracy (%) | {f1_top5:.2f} | {best["val_top5"]:.2f} | {delta_top5:+.2f} |
| Cross-entropy loss | {f1_loss:.4f} | {best["val_loss"]:.4f} | {delta_loss:+.4f} |

Source: `ai/results/f1_per_genre_baseline.csv` + `ai/results/lora_rank_sweep.csv`. Higher top-1/top-5 and lower loss are better.

{real_song_section}## Evaluation data

This adapter is evaluated on two complementary held-out sets, both drawn from the same `val` + `test` splits the LoRA never saw during training:

### 1. Full val split — used for the rank sweep table above

- **Size**: {n_val:,} validation sequences (this genre's val partition)
- **Methodology**: teacher-forced next-token CE / top-1 / top-5 with `pad_id` masking, batch 32, no key augmentation
- **Comparison fairness**: same `evaluate()` call as `ai/results/f1_per_genre_baseline.csv`, same dataloader, same `[GENRE:none]`-initialised embedding-extension setup. Only the LoRA's adapter weights + the 8 new genre embedding rows differ.
- **Output**: `ai/results/lora_rank_sweep.csv` (long format, one row per (genre, rank) cell)

### 2. Curated 130-song real-song eval — used for the **Real-song eval** section below

- **Size**: {eval_set_n} songs from this genre (10 per genre × 13 genres = 130 total)
- **Source partition**: drawn from `splits/val.jsonl` + `splits/test.jsonl` only (no train leakage)
- **Per-genre sources**: {eval_set_sources}
- **Title coverage** (this genre): {eval_set_named} of {eval_set_n} are named real songs; remainder are Chordonomicon entries whose title field is a Spotify track ID by upstream dataset policy
- **Bar range** (this genre): {eval_set_bar_min}–{eval_set_bar_max} bars (≈ {eval_set_avg_dur}s avg at typical tempo for this genre)
- **Build script**: `ai/training/build_eval_real_songs.py --seed 42 --per-genre 10` — deterministic, re-runnable
- **Output**: `ai/results/real_song_eval.csv` (18 models × 130 songs, long format)
- **Full dataset composition + per-source license + methodology**: documented on the released model cards

## License and use

The adapter weights are released under **CC BY-NC 4.0** (matching Chordonomicon, the upstream training corpus). Permitted: research, paper replication, portfolio, demo. Not permitted: commercial deployment without separate licensing of upstream data.

## Usage

Both the base repo and this LoRA repo ship the project's `model.py` and
`tokenizer.py` at the repo root, so external users can load this adapter
end-to-end without cloning anything from GitHub. `snapshot_download`
materializes the full repo on disk; `sys.path` makes the bundled
`model.py` / `tokenizer.py` importable.

Required dependencies: `torch`, `huggingface_hub`, `peft`, `safetensors`.

```python
import sys
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import PeftModel

# 1. Download the base + LoRA repos. Both bundle model.py and tokenizer.py.
base_dir = snapshot_download(repo_id="{BASE_REPO}")
lora_dir = snapshot_download(repo_id="{repo_id}")
sys.path.insert(0, base_dir)  # so the next two imports resolve

from model import MusicTransformer
from tokenizer import ChordTokenizer

# 2. Extended tokenizer (351 base + 8 new genre tokens = 359). The PAD id
#    is unchanged across base and extended tokenizers.
tokenizer = ChordTokenizer(include_extra_genres=True)

# 3. Build the model at the BASE vocab size (351) so F1's state_dict loads
#    cleanly; we grow the embedding rows immediately after. Passing the
#    extended tokenizer's pad_id is safe because PAD is shared (see step 2).
BASE_VOCAB = 351
model = MusicTransformer(
    vocab_size=BASE_VOCAB,
    d_model=512, n_heads=8, d_ff=2048, n_layers=8,
    max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
)
ckpt = torch.load(f"{{base_dir}}/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# 4. Grow token_emb + out_proj from 351 -> 359 (new rows init from
#    [GENRE:none]), then overlay the LoRA's trained extension rows.
def _grow_to_extended_vocab(m, new_vocab, none_id):
    d = m.token_emb.embedding_dim
    new_emb = nn.Embedding(new_vocab, d, padding_idx=m.token_emb.padding_idx)
    with torch.no_grad():
        new_emb.weight[:m.token_emb.num_embeddings] = m.token_emb.weight
        for i in range(m.token_emb.num_embeddings, new_vocab):
            new_emb.weight[i] = m.token_emb.weight[none_id]
    m.token_emb = new_emb
    new_out = nn.Linear(d, new_vocab, bias=False)
    with torch.no_grad():
        new_out.weight[:m.out_proj.out_features] = m.out_proj.weight
        for i in range(m.out_proj.out_features, new_vocab):
            new_out.weight[i] = m.out_proj.weight[none_id]
    m.out_proj = new_out

_grow_to_extended_vocab(model, tokenizer.vocab_size, tokenizer.encode_genre("none"))

ext = torch.load(f"{{lora_dir}}/embedding_extension.pt",
                 map_location="cpu", weights_only=False)
model.token_emb.load_state_dict(ext["token_emb_state"])
model.out_proj.load_state_dict(ext["out_proj_state"])

# 5. Apply the LoRA adapter (the adapter files live at lora_dir/adapter/).
model = PeftModel.from_pretrained(model, f"{{lora_dir}}/adapter")
model.eval()

# 6. Generate a {display_name.lower()} continuation. With LoRA injected,
#    PeftModel.forward routes through the adapted attention layers.
song = {{
    "key": "Cmaj", "time_signature": "4/4", "genre": "{genre}",
    "bars": [["Cmaj7"], ["Fmaj7"]],
}}
prompt_ids = tokenizer.encode_sequence(song)[:-1]
ids = torch.tensor([prompt_ids])
with torch.no_grad():
    for _ in range(32):
        logits = model(ids)  # routed through LoRA via PeftModel
        next_id = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1,
        )
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_id:
            break
print(tokenizer.decode(ids[0].tolist()))
```

## Citation

Cite both the v1 mix-ratio paper (F1 base) and the v2 per-genre LoRA
paper (this adapter family). The v2 arXiv ID will be filled in once the
preprint is posted.

```bibtex
@misc{{lee2026chordmix,
  title         = {{Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation}},
  author        = {{Lee, Jinju}},
  year          = {{2026}},
  eprint        = {{2605.04998}},
  archivePrefix = {{arXiv}}
}}

@misc{{lee2026chordtimeseries,
  title         = {{How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?}},
  author        = {{Lee, Jinju}},
  year          = {{2026}},
  note          = {{arXiv preprint, ID TBD}},
}}
```
"""


def write_card(genre: str, text: str, dry_run: bool) -> Path:
    out = MODEL_DIR / f"ft_f1_lora_{genre}" / "README.md"
    print(f"  card    {out.relative_to(REPO_DIR)}  ({len(text):,} chars)")
    if dry_run:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def upload_to_hf(genre: str, best_rank: int, dry_run: bool) -> None:
    repo_id = f"{ORG}/TheArtist-MusicTransformer-lora-{genre.replace('_', '-')}"
    folder = MODEL_DIR / f"ft_f1_lora_{genre}"
    print(f"  upload  {folder.relative_to(REPO_DIR)} → {repo_id}  (lora_r={best_rank})")
    if dry_run:
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update: best-rank release (lora_r={best_rank}) from rank sweep",
        ignore_patterns=["__pycache__/*", "*.pyc", "*.swp", ".DS_Store"],
    )
    print(f"  → https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + would-stage logs only, no disk writes, no upload")
    ap.add_argument("--plan-only", action="store_true",
                    help="print best-rank plan and exit (no staging, no upload)")
    ap.add_argument("--no-upload", action="store_true",
                    help="stage + write card, but skip HF push")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated genre subset (e.g. 'blues,rock')")
    args = ap.parse_args()

    sweep = load_sweep()
    baseline = load_baseline()
    real_song_eval = load_real_song_eval()
    if real_song_eval:
        print(f"Loaded real-song eval ({len(real_song_eval)} model×genre cells) — injecting into LoRA cards")
    else:
        print("real-song eval not found (eval_real_songs.py not yet run) — LoRA cards will omit that section")
    eval_set_stats = load_real_song_eval_set_per_genre()
    if eval_set_stats:
        print(f"Loaded eval-set composition ({len(eval_set_stats)} genres) — filling per-genre eval data section")
    only = {g.strip() for g in args.only.split(",") if g.strip()}

    plan: list[tuple[str, dict, list[dict]]] = []
    for genre in LORA_GENRES:
        if only and genre not in only:
            continue
        rows = sweep.get(genre, [])
        if not rows:
            print(f"  SKIP {genre:12s}  no sweep data")
            continue
        best = pick_best(rows)
        plan.append((genre, best, rows))

    print(f"\n=== Release plan ({len(plan)} genres) ===")
    print(f"{'genre':<12} {'best_r':>6} {'val_loss':>10} {'val_top1':>10} {'cells':>6}")
    for genre, best, rows in plan:
        ranks_str = "/".join(str(r["rank"]) for r in rows)
        print(f"{genre:<12} {best['rank']:>6d} {best['val_loss']:>10.4f} {best['val_top1']:>10.2f}  ({ranks_str})")

    if args.plan_only:
        return

    for genre, best, rows in plan:
        print(f"\n--- {genre} (lora_r={best['rank']}) ---")
        stage_weights(genre, best["rank"], args.dry_run)
        text = render_card(genre, best, rows, baseline.get(genre, {}),
                            real_song_eval=real_song_eval, eval_set_stats=eval_set_stats)
        write_card(genre, text, args.dry_run)
        if not args.no_upload:
            upload_to_hf(genre, best["rank"], args.dry_run)

    suffix = " (dry-run)" if args.dry_run else (" (no-upload)" if args.no_upload else "")
    print(f"\nDone. {len(plan)} genres processed{suffix}.")


if __name__ == "__main__":
    main()
