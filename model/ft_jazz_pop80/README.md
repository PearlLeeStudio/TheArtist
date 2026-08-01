---
language:
- en
library_name: pytorch
license: cc-by-nc-4.0
pipeline_tag: text-generation
tags:
- music
- music-generation
- chord-generation
- symbolic-music
- music-transformer
- jazz
- pop
---

# TheArtist Music Transformer — F1 (Pop 10K Mix, pop-leaning)

F1 slot of the pop→jazz mix-ratio sweep: the Phase-0 pop baseline fine-tuned on jazz with a 10,000-sequence pop rehearsal buffer, the pop-leaning endpoint. One of six checkpoints released alongside the paper *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026), and the base model the 11 per-genre `lora-*` adapters were trained on and pair with. Recommended for chord-composition workflows targeting pop, rock, CCM, K-pop, J-pop, and modern country with optional jazz coloration; F4 (`ft-pop29`) is the symmetric jazz-leaning endpoint and F3 (`ft-pop50`) the balanced middle.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

> **Weights note.** A full SHA-256 comparison over the 25,841,152-element serialized state dict (the model has 25,661,440 unique parameters; the difference is the tied input/output embedding stored as two tensors) shows the released `best.pt` is weight-identical to the [Phase-0 pop baseline](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline). Best-checkpoint selection used minimum loss on a pop-dominated validation mix, which rose monotonically during jazz fine-tuning, so the pre-fine-tune initialization was retained — and with the run's two-epoch warmup that retained epoch is bit-identical to Phase-0 (a seed/data-dependent outcome, the worst case observed for this seed-42 run, not a general property of the selection rule — matched-data multi-seed retrains selected already-adapted epochs for all three seeds; details on the v2 card).
> The pop/jazz table below therefore reports the run's best-epoch training-log metrics, while the released weights behave exactly like Phase-0 (jazz top-1 72.86 on the same 6-source test).
> A selection-corrected retrain is released as [ft-pop80-v2](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2) (hash-distinct, selected on a jazz-only validation subset). This repository stays as-is: the 11 `lora-*` adapters were trained on and pair with this exact base, so their reported gains are gains over a pure-pop harmonic prior.
> The earlier claim that this base "retains a richer harmonic vocabulary" than the pop baseline is withdrawn.

## Model details

| Field | Value |
|---|---|
| Architecture | Music Transformer with relative positional attention |
| Parameters | 25,661,440 |
| Vocabulary size | 351 tokens |
| Max sequence length | 256 |
| d_model / heads / FFN / layers | 512 / 8 / 2048 / 8 |
| Fine-tune resumed from | Phase-0 pop baseline |
| Best epoch | 6 (training-log; the released `best.pt` is the epoch-3 minimum-mixed-val selection, see the weights note) |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80")
sys.path.insert(0, ckpt_dir)  # so the next two imports resolve

from model import MusicTransformer
from tokenizer import ChordTokenizer

tokenizer = ChordTokenizer()
ckpt = torch.load(f"{ckpt_dir}/best.pt", map_location="cpu", weights_only=False)
model = MusicTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512, n_heads=8, d_ff=2048, n_layers=8,
    max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prompt = ii-V-I in C major; ask for a pop-flavoured continuation.
song = {
    "key": "Cmaj", "time_signature": "4/4", "genre": "pop",
    "bars": [["Dm7", "G7"], ["Cmaj7"]],
}
prompt_ids = tokenizer.encode_sequence(song)[:-1]
ids = torch.tensor([prompt_ids])
with torch.no_grad():
    for _ in range(32):
        logits = model(ids)
        next_id = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1,
        )
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_id:
            break
print(tokenizer.decode(ids[0].tolist()))
```

For per-genre adaptation beyond pop and jazz, see the 11 LoRA adapter repos at [PearlLeeStudio](https://huggingface.co/PearlLeeStudio) — they chain on top of this base.

## Evaluation

**Held-out per-genre test sets** — the figures below are the fine-tuning run's best-epoch (epoch 6) training-log metrics, with the jazz column measured on the 6-source jazz test (167 sequences); the released weights do not embody them (see the weights note above).

| Metric | Pop test | Jazz test (6-src) |
|---|---:|---:|
| Top-1 accuracy | **84.60%** | 81.03% |
| Top-5 accuracy | 96.96% | 92.41% |
| Perplexity | 1.78 | 2.31 |
| Δ vs. Phase-0 baseline | **+0.39** | +8.17 |

### Out-of-distribution per-genre baseline

F1 alone (no LoRA) on the 11 per-genre val splits the LoRA adapters target (10 Chordonomicon genre subsets plus the Bach-chorale classical split). The eight genres beyond the base vocabulary are encoded at `[GENRE:none]`-initialised embedding rows (effectively unconditioned); rock, blues, and bossa use their existing base-vocab `[GENRE:X]` tokens. This is the no-LoRA reference reported on every `lora-<genre>` adapter card — per the weights note, these are measurements of the released weights and are cleanly interpretable as a pure-pop prior.

| Genre | Val seq. | F1 top-1 (%) | F1 top-5 (%) | F1 val loss |
|---|---:|---:|---:|---:|
| hip-hop    | 1,402 | 86.51 | 96.27 | 0.6240 |
| electronic | 1,519 | 84.50 | 95.93 | 0.6835 |
| rock       | 4,891 | 82.79 | 96.75 | 0.5865 |
| folk       | 6,075 | 82.66 | 95.80 | 0.7406 |
| funk       | 283   | 82.54 | 94.38 | 0.7878 |
| country    | 6,173 | 82.45 | 96.22 | 0.7402 |
| rnb/soul   | 955   | 82.09 | 94.12 | 0.8119 |
| blues      | 994   | 81.70 | 94.80 | 0.8137 |
| gospel     | 374   | 79.34 | 94.73 | 0.8813 |
| bossa      | 1,431 | 78.33 | 93.64 | 0.9635 |
| classical  | 37    | 43.54 | 72.82 | 2.8653 |

<!-- real-song-eval:begin -->
### Per-genre real-song eval

On this eval set the released F1 base peaks on **hip_hop** (90.66%) and struggles most on **classical** (49.55%). The 11 per-genre LoRA adapters ([sister `lora-*` repos](https://huggingface.co/PearlLeeStudio)) are the recommended path beyond pop and jazz — for the eight genres without a `[GENRE:X]` token in the 351-token vocabulary (all but rock, blues, and bossa) the base decodes without genre conditioning here.

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 86.68 | 96.01 | 0.5734 |
| rock | 10 | 86.69 | 97.48 | 0.4578 |
| jazz | 10 | 64.96 | 81.16 | 1.8958 |
| blues | 10 | 81.52 | 93.91 | 0.8410 |
| bossa | 10 | 81.43 | 95.47 | 0.7825 |
| classical | 10 | 49.55 | 81.17 | 2.2389 |
| country | 10 | 85.90 | 98.44 | 0.5152 |
| electronic | 10 | 87.39 | 98.45 | 0.5072 |
| folk | 10 | 85.04 | 98.92 | 0.5244 |
| funk | 10 | 83.85 | 96.03 | 0.6811 |
| gospel | 10 | 79.79 | 96.85 | 0.7367 |
| hip_hop | 10 | 90.66 | 98.59 | 0.3957 |
| rnb_soul | 10 | 85.10 | 97.07 | 0.5877 |

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy). Full composition:

| Genre | n | Source(s) | Bar range | Avg duration · named |
|---|---:|---|---|---|
| pop | 10 | billboard | 58–116 | 189s · 10/10 named |
| rock | 10 | chordonomicon_rock | 52–87 | 127s · 0/10 named |
| jazz | 10 | choco:jazz-corpus, choco:real-book, jazzstandards, jht | 16–89 | 72s · 10/10 named |
| blues | 10 | chordonomicon_blues | 24–46 | 93s · 0/10 named |
| bossa | 10 | chordonomicon_bossa | 24–78 | 88s · 0/10 named |
| classical | 10 | chordonomicon_classical | 11–40 | 60s · 10/10 named |
| country | 10 | chordonomicon_country | 30–81 | 110s · 0/10 named |
| electronic | 10 | chordonomicon_electronic | 25–84 | 89s · 0/10 named |
| folk | 10 | chordonomicon_folk | 33–82 | 114s · 0/10 named |
| funk | 10 | chordonomicon_funk | 30–60 | 92s · 0/10 named |
| gospel | 10 | chordonomicon_gospel | 24–85 | 98s · 0/10 named |
| hip_hop | 10 | chordonomicon_hip_hop | 24–81 | 136s · 0/10 named |
| rnb_soul | 10 | chordonomicon_rnb_soul | 34–82 | 128s · 0/10 named |

**Source license summary**: McGill Billboard (CC0, named pop songs), Jazz Harmony Treebank / JazzStandards / WJazzD (Public / community-redistributed, named jazz standards), Bach chorales via music21 (public domain, named pieces), Chordonomicon per-genre subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream dataset policy — progressions are real songs).
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences (Jazz Harmony Treebank, JazzStandards, Weimar Jazz Database, JAAH) plus 10,000 pop rehearsal sequences sub-sampled with seed 42 from the Phase-0 pop training split — pop:jazz ≈ 6.6:1 in the mix. Fine-tune hyperparameters: peak learning rate 2 × 10⁻⁵, two-epoch warmup, ten epochs maximum with patience 5.

## License

**CC BY-NC 4.0** (weights; matching Chordonomicon, the dominant training corpus). Research, paper replication, portfolio, and demo use are permitted; commercial use is not.

## Citation

```bibtex
@misc{lee2026chordmix,
  title         = {Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2605.04998},
  archivePrefix = {arXiv}
}

@misc{lee2026chordtimeseries,
  title         = {How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2606.07334},
  archivePrefix = {arXiv}
}
```
