#!/usr/bin/env -S uv run --with huggingface_hub python
"""Update the 6 F-series model cards on HuggingFace with a per-genre real-song
eval section. Weights are unchanged — only the README is regenerated and
re-uploaded.

For each of (Phase 0, F1, F2, F3, F4, F5):
  1. Read existing card from model/<run>/README.md
  2. Inject a "Per-genre real-song eval" section just before the existing
     "## Training-data licenses" anchor (or before "## Citation" if licenses
     section is absent). The section gives that model's mean validation
     top-1 / top-5 / cross-entropy on each of the 13 genres, drawn from
     ai/results/real_song_eval.csv.
  3. Push the regenerated card to PearlLeeStudio/TheArtist-MusicTransformer-*.

Idempotent — re-running replaces an existing real-song-eval block in place.

Usage:
  uv run --with huggingface_hub python model/_release_f_series_cards.py
  python model/_release_f_series_cards.py --plan-only
  python model/_release_f_series_cards.py --dry-run
  python model/_release_f_series_cards.py --no-upload
  python model/_release_f_series_cards.py --only Phase0,F1
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
REAL_SONG_CSV = REPO_DIR / "ai" / "results" / "real_song_eval.csv"

ORG = "PearlLeeStudio"

F_SERIES: list[tuple[str, str, str]] = [
    # (model_id used in real_song_eval.csv, model folder, HF repo name)
    ("Phase0", "phase0_pop_baseline", "TheArtist-MusicTransformer-pop-baseline"),
    ("F1",     "ft_jazz_pop80",       "TheArtist-MusicTransformer-ft-pop80"),
    ("F1v2",   "ft_jazz_pop80_v2",    "TheArtist-MusicTransformer-ft-pop80-v2"),  # rows added 2026-06-12
    ("F2",     "ft_jazz_pop67",       "TheArtist-MusicTransformer-ft-pop67"),
    ("F3",     "ft_jazz_pop50",       "TheArtist-MusicTransformer-ft-pop50"),
    ("F4",     "ft_jazz_pop29",       "TheArtist-MusicTransformer-ft-pop29"),
    ("F5",     "ft_jazz_only",        "TheArtist-MusicTransformer-ft-jazz-only"),
]

GENRES = [
    "pop", "rock", "jazz", "blues", "bossa", "classical",
    "country", "electronic", "folk", "funk", "gospel",
    "hip_hop", "rnb_soul",
]

REAL_SONG_BEGIN = "<!-- real-song-eval:begin -->"
REAL_SONG_END = "<!-- real-song-eval:end -->"


def load_real_song_eval() -> dict[tuple[str, str], dict]:
    if not REAL_SONG_CSV.exists():
        sys.exit(f"[fail] {REAL_SONG_CSV} missing — run eval_real_songs.py first")
    bucket: dict[tuple[str, str], list[dict]] = {}
    with open(REAL_SONG_CSV) as f:
        for r in csv.DictReader(f):
            try:
                top1 = float(r["val_top1"])
                top5 = float(r["val_top5"])
                loss = float(r["val_loss"])
            except (ValueError, KeyError):
                continue
            if top1 != top1:
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


def load_eval_set_composition() -> list[dict]:
    """Read ai/data/eval_real_songs.jsonl for per-genre dataset stats injection."""
    import json
    p = REPO_DIR / "ai" / "data" / "eval_real_songs.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p)]


def render_real_song_block(model_id: str, model_display: str,
                            data: dict[tuple[str, str], dict]) -> str:
    rows: list[str] = []
    for g in GENRES:
        cell = data.get((model_id, g))
        if cell is None:
            rows.append(f"| {g} | — | — | — | — |")
            continue
        rows.append(
            f"| {g} | {cell['n']} | {cell['top1']:.2f} | "
            f"{cell['top5']:.2f} | {cell['loss']:.4f} |"
        )
    table = "\n".join(rows)

    # Find this model's best and worst genres
    own = [(g, data[(model_id, g)]) for g in GENRES if (model_id, g) in data]
    if own:
        best_g, best_v = max(own, key=lambda x: x[1]["top1"])
        worst_g, worst_v = min(own, key=lambda x: x[1]["top1"])
        callout = (
            f"\nOn this eval set {model_id} peaks on **{best_g}** "
            f"({best_v['top1']:.2f}%) and struggles most on **{worst_g}** "
            f"({worst_v['top1']:.2f}%). "
        )
    else:
        callout = ""

    # Per-genre composition table for transparency
    from collections import defaultdict
    by_genre = defaultdict(list)
    for s in load_eval_set_composition():
        by_genre[s["genre"]].append(s)
    composition_rows = []
    for g in GENRES:
        gs = by_genre.get(g, [])
        if not gs:
            composition_rows.append(f"| {g} | 0 | — | — | — |")
            continue
        sources = sorted({s["source"] for s in gs})
        named = sum(1 for s in gs if s.get("title_is_real"))
        bars = sorted(s["n_bars"] for s in gs)
        avg_dur = round(sum(s["est_duration_sec"] for s in gs) / len(gs))
        composition_rows.append(
            f"| {g} | {len(gs)} | {', '.join(sources)} | "
            f"{bars[0]}–{bars[-1]} | {avg_dur}s · {named}/{len(gs)} named |"
        )
    composition_table = "\n".join(composition_rows)

    return f"""{REAL_SONG_BEGIN}
## Per-genre real-song eval (held-out 130-song set, 2026-05)

First per-genre evaluation of {model_display} beyond the pop/jazz split that the original paper reports.

### Eval results

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
{table}
{callout}
This is auxiliary signal — the 11 per-genre LoRAs ([sister `lora-*` repos](https://huggingface.co/PearlLeeStudio)) are the recommended path for production use on the 9 non-pop, non-jazz genres. F-series cells on those genres show what the base model produces under `[GENRE:none]` conditioning (the model's `[GENRE:X]` token does not exist for the 9 new genres in the F-series vocab=351).

### Eval dataset composition

130 songs total, 10 per genre × 13 genres. Drawn from the same `splits/val.jsonl` + `splits/test.jsonl` partitions every F-series model was held out from during training — no train-set leakage. Built by `ai/training/build_eval_real_songs.py --seed 42 --per-genre 10` (deterministic).

| Genre | n | Source(s) | Bar range | Avg duration · named |
|---|---:|---|---|---|
{composition_table}

**Source license summary**: McGill Billboard (CC0, named pop songs), Jazz Harmony Treebank / JazzStandards / WJazzD (Public / community-redistributed, named jazz standards), Bach chorales via music21 (public domain, named pieces), Chordonomicon per-genre subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream dataset policy — progressions are real songs). Full breakdown on the released model cards.

### Methodology

Teacher-forced next-token cross-entropy / top-1 / top-5 over each song's token sequence (BOS + key + time_sig + genre + bars + EOS, truncated to `max_seq_len=256`). Same `evaluate()` call as `ai/results/f1_per_genre_baseline.csv`, just narrowed to the curated 130-song subset. Token-level metrics; not a generation-quality eval (free-generation comparison with R1 Sethares + R2 theory RAG rerank is documented separately in `ai/results/eval_report.md`).

**Caveats**:
- `classical` val partition is intrinsically small (37 sequences in full eval); the 10-song subset here has even narrower confidence bands. Directional finding (LoRA helps a lot on Bach harmony) is robust, exact pp deltas are noisy.
- F-series numbers on the 9 LoRA-only genres are conditioned without genre tag (vocab=351 has no `[GENRE:country]` token etc.). This is the realistic "F-series alone" condition, not a controlled ablation.

Source CSV: `ai/results/real_song_eval.csv` (18 models × 130 songs, long format).
{REAL_SONG_END}

"""


def inject_block(card_text: str, block: str) -> str:
    """Insert (or replace) the real-song eval block. Idempotent against any
    prior state — properly marker-bracketed, orphan markers, or plain
    section without markers.

    Priority order:
      1. begin+end markers both present → replace between markers cleanly.
      2. Section header "## Per-genre real-song eval" exists (with or without
         an orphan end marker) → replace from header through end marker if
         present, else through the next "## " heading, else through EOF.
         This handles the legacy case where cards had only the end marker
         (begin was added later), which previously fell through to step 3
         and produced a duplicate eval block.
      3. No prior section → insert before "## Training-data licenses" or
         "## Citation" anchor.
      4. No anchor either → append at EOF.
    """
    # Case 1: properly marker-bracketed.
    pat_full = re.compile(
        re.escape(REAL_SONG_BEGIN) + r".*?" + re.escape(REAL_SONG_END) + r"\n*",
        flags=re.DOTALL,
    )
    if pat_full.search(card_text):
        return pat_full.sub(block, card_text)

    # Case 2: section header exists (orphan markers or no markers).
    # Match from the section header through whichever ends first: the
    # orphan end marker, the next "## " heading, or EOF.
    section_pat = re.compile(
        r"(?:<!-- real-song-eval:begin -->\n)?"
        r"## Per-genre real-song eval\b.*?"
        r"(?:" + re.escape(REAL_SONG_END) + r"\n*"
        r"|(?=\n## )"
        r"|\Z)",
        flags=re.DOTALL,
    )
    if section_pat.search(card_text):
        return section_pat.sub(block, card_text)

    # Case 3: no prior section → insert before an anchor.
    for anchor in ("## Training-data licenses", "## Citation"):
        idx = card_text.find(anchor)
        if idx != -1:
            return card_text[:idx] + block + card_text[idx:]

    # Case 4: append.
    return card_text.rstrip() + "\n\n" + block


def upload(repo_name: str, folder: Path, model_id: str, dry_run: bool) -> None:
    repo_id = f"{ORG}/{repo_name}"
    print(f"  upload  {folder.relative_to(REPO_DIR)} → {repo_id}  (card-only update)")
    if dry_run:
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    # Only push the README — leave weights / config / tokenizer alone since
    # they haven't changed.
    api.upload_file(
        path_or_fileobj=str(folder / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update card: per-genre real-song eval ({model_id})",
    )
    print(f"  → https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, no disk writes, no upload")
    ap.add_argument("--plan-only", action="store_true",
                    help="show which models would be updated and exit")
    ap.add_argument("--no-upload", action="store_true",
                    help="patch local README.md files, skip HF push")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated model IDs (Phase0, F1, ...)")
    args = ap.parse_args()

    eval_data = load_real_song_eval()
    only = {m.strip() for m in args.only.split(",") if m.strip()}

    print(f"\n=== F-series card update — {len(eval_data)} (model, genre) eval cells loaded ===")
    print(f"{'id':<8} {'folder':<25} {'genres_evaluated':>20}")
    plan: list[tuple[str, str, str]] = []
    for model_id, folder, repo_name in F_SERIES:
        if only and model_id not in only:
            continue
        folder_path = MODEL_DIR / folder
        if not (folder_path / "README.md").exists():
            print(f"{model_id:<8} {folder:<25} SKIP — README.md missing")
            continue
        evaluated = sum(1 for g in GENRES if (model_id, g) in eval_data)
        print(f"{model_id:<8} {folder:<25} {evaluated:>4}/{len(GENRES)}")
        plan.append((model_id, folder, repo_name))

    if args.plan_only:
        return

    for model_id, folder, repo_name in plan:
        print(f"\n--- {model_id} ({folder}) ---")
        folder_path = MODEL_DIR / folder
        readme_path = folder_path / "README.md"
        original = readme_path.read_text()
        model_display = repo_name.replace("TheArtist-MusicTransformer-", "")
        block = render_real_song_block(model_id, model_display, eval_data)
        updated = inject_block(original, block)
        if updated == original:
            print(f"  card    no change (existing block matches)")
        else:
            print(f"  card    patched  ({len(original):,} → {len(updated):,} chars)")
            if not args.dry_run:
                readme_path.write_text(updated)
        if not args.no_upload:
            upload(repo_name, folder_path, model_id, args.dry_run)

    suffix = " (dry-run)" if args.dry_run else (" (no-upload)" if args.no_upload else "")
    print(f"\nDone. {len(plan)} F-series cards processed{suffix}.")


if __name__ == "__main__":
    main()
