#!/usr/bin/env python3
"""Generate the 11 per-genre LoRA model cards from data.

Every card follows the standard model-card layout shared by all 18 cards:

    frontmatter, title, description, links line
    ## Model details    base, adapter, trainable parameters, vocabulary, files
    ## Usage            end-to-end loader (bundled model.py / tokenizer.py)
    ## Evaluation       val-split table, rank sweep, real-song table
    ## Training data    source subset and split
    ## Limitations      scope, metric caveats, genre-specific reading
    ## License
    ## Citation

Every table number is computed at generation time from the released artifacts
and result CSVs. The genre-specific Limitations bullets are fixed text that
quotes arXiv:2606.07334v5 (header-free rescoring against F1 given the pop
marker), so they change only when the paper does. Sources:

    ai/results/lora_rank_sweep.csv        rank sweep + released-rank val metrics
    ai/results/f1_per_genre_baseline.csv  F1-base column (same val split/dataloader)
    ai/results/real_song_eval.csv         10-song real-song eval (mean per genre)
    ai/checkpoints/ft_f1_lora_<g>[_rN]/adapter/adapter_config.json  r / alpha / dropout
    ai/checkpoints/.../adapter/adapter_model.safetensors            file size

Usage:  python3 -I model/_generate_lora_cards.py
Push:   uv run python model/_patch_readmes.py --only-loras --message "..."
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
CKPT_DIR = REPO_DIR / "ai" / "checkpoints"
RESULTS = REPO_DIR / "ai" / "results"

BASE_REPO = "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80"
FULL_FT_PARAMS = 25_849_344  # full-fine-tune trainable set as trained (arXiv:2606.07334v5 parameter table)
EMB_PARAMS = 367_616  # full token_emb + out_proj kept trainable (359 * 512 * 2), per lora_train.py
V5 = "[arXiv:2606.07334](https://arxiv.org/abs/2606.07334) v5"

# Unique song ids across train/val/test of the canonical splits (long songs are
# windowed into ~1.15x as many sequences, and GENRE_INFO counts are sequences).
UNIQUE_SONGS = {
    "blues": 8_728, "bossa": 12_011, "classical": 237, "country": 53_236,
    "electronic": 13_148, "folk": 52_865, "funk": 2_335, "gospel": 2_856,
    "hip_hop": 11_584, "rnb_soul": 8_217, "rock": 42_781,
}

# genre -> (display name, subset sequences, one-line character)
GENRE_INFO: dict[str, tuple[str, int, str]] = {
    "country":    ("Country",     61_736, "country pop, country rock, and modern country idioms"),
    "funk":       ("Funk",         2_837, "funk grooves with extended dominants and altered sevenths"),
    "gospel":     ("Gospel",       3_747, "gospel cadences, plagal motion, and gospel substitutions"),
    "rnb_soul":   ("R&B / Soul",   9_550, "R&B and soul harmony with subdominant minor, suspensions, and modal interchange"),
    "hip_hop":    ("Hip-Hop",     14_021, "hip-hop chord conventions with minor-key, neo-soul-adjacent harmony"),
    "electronic": ("Electronic",  15_196, "electronic harmony with minor keys, repetitive vamps, and tonic-centred motion"),
    "folk":       ("Folk",        60_752, "folk and singer-songwriter harmony"),
    "classical":  ("Classical",      371, "Bach chorales"),
    "rock":       ("Rock",        48_910, "classic rock and modern rock harmony"),
    "blues":      ("Blues",        9_944, "12-bar and modified blues forms"),
    "bossa":      ("Bossa nova",  14_315, "Brazilian bossa nova jazz harmony"),
}

# genre -> real-song eval-set facts (source key, bar range, titled?), matching
# the published composition table on the F-series cards.
REAL_SONG_FACTS: dict[str, tuple[str, str, bool]] = {
    "country":    ("chordonomicon_country",    "30–81", False),
    "funk":       ("chordonomicon_funk",       "30–60", False),
    "gospel":     ("chordonomicon_gospel",     "24–85", False),
    "rnb_soul":   ("chordonomicon_rnb_soul",   "34–82", False),
    "hip_hop":    ("chordonomicon_hip_hop",    "24–81", False),
    "electronic": ("chordonomicon_electronic", "25–84", False),
    "folk":       ("chordonomicon_folk",       "33–82", False),
    "classical":  ("chordonomicon_classical",  "11–40", True),
    "rock":       ("chordonomicon_rock",       "52–87", False),
    "blues":      ("chordonomicon_blues",      "24–46", False),
    "bossa":      ("chordonomicon_bossa",      "24–78", False),
}

TITLED_NOTE = "All 10 are named Bach chorales."
ANON_NOTE = ("Their titles are Spotify track IDs by upstream Chordonomicon policy,"
             " but the progressions are real songs.")

SOURCE_CHORDONOMICON = (
    "the **{g}** subset of the [Chordonomicon dataset]"
    "(https://huggingface.co/datasets/ailsntua/Chordonomicon)"
)
SOURCE_CLASSICAL = (
    "**Bach chorales** curated from the public-domain "
    "[music21](https://web.mit.edu/music21/) corpus"
)
# arXiv:2606.07334v5, Sections 5.3, 6.1, 6.3 and 6.7. Released adapters that
# predict the placeholder C-major key (v5 Section 5.3): blues, bossa, country,
# gospel. Blues and bossa markers are in the 351-token vocab but never an input
# in Phase-0 training.
PREDICTS_C_MAJOR = {"blues", "bossa", "country", "gospel"}
BASE_VOCAB_MARKERS = {"blues", "bossa"}

HEADER_LABEL = "- **Header tokens.** Part of each Δ comes from header-token predictions:"
RANK_STEP = (" In the rank sweep, a rank about one key-token step above the others"
             " is a run that predicted the C-major key.")


def genre_bullets(genre: str) -> str:
    """Two Limitations bullets on how much of the Δ is the header tokens and the marker."""
    tok = f"`[GENRE:{genre}]`"
    if genre == "classical":
        return (
            f"{HEADER_LABEL} F1 does not predict the {tok} marker, and this adapter"
            " does. It also gets all 38 time-signature targets of the Bach-chorale"
            " test set right where F1 gets 18.\n"
            f"- **Bach chorales gain beyond the marker.** F1 has never read {tok} as"
            " an input in training, but unlike in the ten Chordonomicon-derived"
            " genres the adapter adds more than a usable marker: scored without the"
            " header targets and with F1 given `[GENRE:pop]` instead, LoRA still"
            f" leads by 9.6 pp, and by 10.2 pp on the real-song check ({V5}). Bach"
            " chorale is also the only genre whose rank sweep shows a clear rank"
            " trend."
        )
    if genre == "rock":
        return (
            f"{HEADER_LABEL} F1 does not predict the {tok} marker, and this adapter"
            f" does.{RANK_STEP}\n"
            "- **Rock gains about nothing beyond the header tokens.** Rock is the one target"
            " marker F1 has read in training (Phase 0 maps the rock, alternative,"
            " punk, and metal tags to it), so without the header targets rock gains"
            " +0.01 to +0.06 pp for the methods other than BitFit"
            f" ({V5})."
        )
    header = (f"F1 predicts neither the {tok} marker nor the placeholder C-major key,"
              " and this adapter predicts both, the key alone adding about one"
              " key-token step to the Δ."
              if genre in PREDICTS_C_MAJOR else
              f"F1 does not predict the {tok} marker, and this adapter does.")
    origin = ("it is in the base vocabulary, but Phase 0 maps every Chordonomicon"
              " tag onto pop, rock, or jazz"
              if genre in BASE_VOCAB_MARKERS else
              "the eight new markers start as copies of the equally unused"
              " `[GENRE:none]` row")
    return (
        f"{HEADER_LABEL} {header}{RANK_STEP}\n"
        "- **The gain is mostly a usable genre marker.** F1 has never read"
        f" {tok} as an input in training ({origin}). Scored without the header"
        " targets and with F1 given `[GENRE:pop]` instead, the released matched"
        " adapters gain −0.02 to +0.12 pp in the ten non-chorale genres"
        f" ({V5}), so outside Bach chorales this adapter mainly adds a genre"
        " marker the model can use."
    )


TEMPLATE = """---
base_model: PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80
language:
- en
library_name: peft
license: cc-by-nc-4.0
pipeline_tag: text-generation
tags:
- music
- music-generation
- chord-generation
- symbolic-music
- music-transformer
- lora
- peft
- {genre_tag}
---

# TheArtist Music Transformer — LoRA Adapter ({display_name})

LoRA adapter that steers the F1 base toward **{display_name_lower} chord progressions**: {genre_blurb}. It is one of eleven per-genre adapters from the paper [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026), all trained on and loaded over the same F1 base. Use it to generate {display_name_lower} progressions or to compare genres over one base.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Web app](https://artist.pearlleestudio.co) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

## Model details

| Field | Value |
|---|---|
| Base model | [{base_repo}](https://huggingface.co/{base_repo}) (F1). The released F1 is weight-identical to the Phase-0 pop baseline (a checkpoint-selection artifact). |
| Adapter | LoRA on the Q/K/V projections (`w_q`, `w_k`, `w_v`), r={lora_r} (best of r ∈ {{4, 8, 16, 32, 64}}), α={lora_alpha}, dropout {lora_dropout} |
| Trainable parameters | {params:,} ({lora_params:,} LoRA + {emb_params:,} embedding and output), {pct:.1f}% of the 25,849,344 that full fine-tuning updates |
| Vocabulary | 359 tokens (351 base + 8 new genre tokens) |
| Files | `adapter/` (LoRA weights, {size_mb} MB) and `embedding_extension.pt` (the trained embedding and output matrices) |

## Usage

Requires `torch`, `huggingface_hub`, `peft`, `safetensors`. Both repos bundle `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import PeftModel

# 1. Download the base + LoRA repos. Both bundle model.py and tokenizer.py.
base_dir = snapshot_download(repo_id="{base_repo}")
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

# 6. Generate a {display_name_lower} continuation. With LoRA injected,
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

## Evaluation

All tables report teacher-forced next-token metrics. Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker).

On this genre's val split ({val_seq:,} sequences, no key augmentation) both columns use the same dataloader and the same `[GENRE:none]`-initialised embedding extension, so only the adapter weights and the trained embedding rows differ.

| Metric | F1 base | F1 + this LoRA (r={lora_r}) | Δ |
|---|---:|---:|---:|
| Top-1 accuracy (%) | {f1_top1:.2f} | **{lora_top1:.2f}** | **{d_top1:+.2f}** |
| Top-5 accuracy (%) | {f1_top5:.2f} | {lora_top5:.2f} | {d_top5:+.2f} |
| Cross-entropy loss | {f1_loss:.4f} | {lora_loss:.4f} | {d_loss:+.4f} |

Rank sweep the released adapter was selected from (minimum val loss, top-1 tiebreak):

| Rank | val_loss | Top-1 (%) | Δ Top-1 vs F1 |
|---|---:|---:|---:|
{rank_rows}

Real-song check, mean over **10 held-out {display_name_lower} songs** of {real_bars} bars, this genre's slice of a 130-song eval set (10 per genre × 13 genres, seed 42) drawn only from the held-out val/test partitions. {real_note}

| Model | Top-1 (%) | Top-5 (%) | Loss |
|---|---:|---:|---:|
| F1 base | {rs_f1_top1:.2f} | {rs_f1_top5:.2f} | {rs_f1_loss:.4f} |
| **F1 + this LoRA** | **{rs_lora_top1:.2f}** | {rs_lora_top5:.2f} | {rs_lora_loss:.4f} |
| Δ | {rs_d_top1:+.2f} | {rs_d_top5:+.2f} | {rs_d_loss:+.4f} |

## Training data

{count:,} sequences ({unique:,} songs) from {source_phrase}, with a song-level 80/10/10 split (seed 42) and 12-key augmentation on train. The adapter trains on this subset alone, and the best checkpoint has the minimum val loss.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
{genre_bullets}

## License

**CC BY-NC 4.0** (matching Chordonomicon, the upstream training corpus). Research, paper replication, portfolio, and demo use are permitted; commercial use is not.

## Citation

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
  eprint        = {{2606.07334}},
  archivePrefix = {{arXiv}}
}}

```
"""


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def main() -> None:
    sweep = load_csv(RESULTS / "lora_rank_sweep.csv")
    base = {r["genre"]: r for r in load_csv(RESULTS / "f1_per_genre_baseline.csv")}
    real = load_csv(RESULTS / "real_song_eval.csv")

    written = 0
    for genre, (display, count, blurb) in GENRE_INFO.items():
        rows = sorted((r for r in sweep
                       if r["genre"] == genre and r["rank"].isdigit()),
                      key=lambda r: int(r["rank"]))
        if not rows:
            print(f"  skip {genre}: no rank-sweep rows")
            continue
        best = min(rows, key=lambda r: (float(r["val_loss"]), -float(r["val_top1"])))
        b = base[genre]

        # released artifact facts
        rdir = CKPT_DIR / f"ft_f1_lora_{genre}_r{best['rank']}"
        if not rdir.exists():
            rdir = CKPT_DIR / f"ft_f1_lora_{genre}"
        acfg = json.loads((rdir / "adapter" / "adapter_config.json").read_text())
        size = (rdir / "adapter" / "adapter_model.safetensors").stat().st_size
        lora_r = int(acfg["r"])
        lora_params = 24_576 * lora_r   # 24 Q/K/V pairs: 8 layers x 3 proj x 2*512*r (= the safetensors count)
        params = lora_params + EMB_PARAMS

        rank_rows = "\n".join(
            f"| {'**r=' + r['rank'] + '**' if r is best else 'r=' + r['rank']} "
            f"| {float(r['val_loss']):.4f} | {float(r['val_top1']):.2f} "
            f"| {float(r['dval_top1_vs_baseline']):+.2f}"
            f"{' ← released' if r is best else ''} |"
            for r in rows
        )

        def rs(model: str) -> tuple[float, float, float]:
            sel = [r for r in real if r["model"] == model and r["genre"] == genre]
            assert len(sel) == 10, f"{model}/{genre}: {len(sel)} rows"
            return (mean([float(r["val_top1"]) for r in sel]),
                    mean([float(r["val_top5"]) for r in sel]),
                    mean([float(r["val_loss"]) for r in sel]))

        rs_f1, rs_lora = rs("F1"), rs(f"LoRA-{genre}")
        _src, bars, titled = REAL_SONG_FACTS[genre]
        classical = genre == "classical"

        text = TEMPLATE.format(
            genre=genre,
            genre_tag=genre.replace("_", "-"),
            display_name=display,
            display_name_lower=display.lower() if genre != "rnb_soul" else "R&B / soul",
            genre_blurb=blurb,
            base_repo=BASE_REPO,
            repo_id=f"PearlLeeStudio/TheArtist-MusicTransformer-lora-{genre.replace('_', '-')}",
            count=count,
            unique=UNIQUE_SONGS[genre],
            source_phrase=(SOURCE_CLASSICAL if classical
                           else SOURCE_CHORDONOMICON.format(g=display.lower())),
            lora_r=lora_r,
            lora_alpha=int(acfg["lora_alpha"]),
            lora_dropout=acfg["lora_dropout"],
            pct=100 * params / FULL_FT_PARAMS,
            lora_params=lora_params,
            emb_params=EMB_PARAMS,
            params=params,
            size_mb=f"{size / 1_048_576:.1f}",
            val_seq=int(best["val_sequences"]),
            f1_top1=float(b["val_top1"]), f1_top5=float(b["val_top5"]),
            f1_loss=float(b["val_loss"]),
            lora_top1=float(best["val_top1"]), lora_top5=float(best["val_top5"]),
            lora_loss=float(best["val_loss"]),
            d_top1=float(best["val_top1"]) - float(b["val_top1"]),
            d_top5=float(best["val_top5"]) - float(b["val_top5"]),
            d_loss=float(best["val_loss"]) - float(b["val_loss"]),
            rank_rows=rank_rows,
            rs_f1_top1=rs_f1[0], rs_f1_top5=rs_f1[1], rs_f1_loss=rs_f1[2],
            rs_lora_top1=rs_lora[0], rs_lora_top5=rs_lora[1], rs_lora_loss=rs_lora[2],
            rs_d_top1=rs_lora[0] - rs_f1[0], rs_d_top5=rs_lora[1] - rs_f1[1],
            rs_d_loss=rs_lora[2] - rs_f1[2],
            real_bars=bars,
            real_note=TITLED_NOTE if titled else ANON_NOTE,
            genre_bullets=genre_bullets(genre),
        )
        out = MODEL_DIR / f"ft_f1_lora_{genre}" / "README.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(text, encoding="utf-8")
        written += 1
        print(f"  wrote {out.relative_to(REPO_DIR)} (r={lora_r})")
    print(f"\ngenerated {written} LoRA model cards")


if __name__ == "__main__":
    main()
