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

# TheArtist Music Transformer — F3 (Pop 2.5K Mix)

Jazz-adapted chord model fine-tuned from the Phase-0 pop baseline with a 2,500-sequence pop rehearsal buffer, the middle of the mix-ratio sweep and one of six checkpoints from [Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation](https://arxiv.org/abs/2605.04998) (Lee, 2026). Its continuations introduce secondary dominants, chromatic passing diminished chords, and other jazz voice-leading vocabulary that the Phase-0 baseline does not produce. The choice among the rehearsal runs, all usable for pop and jazz, is stylistic: F3 sits at an intermediate register, while F1 (`ft-pop80`) commits more to pop and F4 (`ft-pop29`) to jazz.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Web app](https://artist.pearlleestudio.co) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

## Model details

| Field | Value |
|---|---|
| Architecture | Music Transformer with relative positional attention |
| Parameters | 25,661,440 |
| Vocabulary size | 351 tokens |
| Max sequence length | 256 |
| d_model / heads / FFN / layers | 512 / 8 / 2048 / 8 |
| Fine-tune resumed from | Phase-0 pop baseline |
| Best epoch (paper) | 9 |
| Released weights | Epoch 4, the minimum-mixed-validation-loss checkpoint (pop 84.06 / jazz 79.77 top-1 on the same test sets), hash-distinct from Phase 0 and from the released [F1](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80), which is weight-identical to Phase 0 |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50")
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

# Prompt = ii-V-I in C major; ask for a jazz-flavoured continuation.
song = {
    "key": "Cmaj", "time_signature": "4/4", "genre": "jazz",
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

For per-genre adaptation beyond pop and jazz, see the 11 LoRA adapter repos at [PearlLeeStudio](https://huggingface.co/PearlLeeStudio) — they chain on the released `ft-pop80` (F1) base, not this checkpoint.

## Evaluation

Held-out pop and jazz test sets, the paper's tabulated epoch 9 (best jazz, Table 4):

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | 84.20% | **80.99%** |
| Top-5 accuracy | 96.87% | 92.63% |
| Perplexity | 1.82 | 2.29 |
| Δ vs. Phase-0 baseline | −0.01 | +8.13 |
| Δ vs. Phase-0, header targets excluded | −0.90 | +7.63 |

Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker). The last row leaves them out ([arXiv:2605.04998](https://arxiv.org/abs/2605.04998) v4, Table 6).

<!-- real-song-eval:begin -->
### Per-genre real-song eval

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 86.31 | 95.80 | 0.5821 |
| rock | 10 | 87.05 | 97.40 | 0.4669 |
| jazz | 10 | 71.01 | 86.70 | 1.3335 |
| blues | 10 | 81.86 | 93.90 | 0.8056 |
| bossa | 10 | 82.02 | 95.81 | 0.7311 |
| classical | 10 | 49.73 | 83.61 | 2.1032 |
| country | 10 | 85.96 | 98.24 | 0.5182 |
| electronic | 10 | 87.02 | 98.29 | 0.5093 |
| folk | 10 | 84.80 | 98.70 | 0.5285 |
| funk | 10 | 83.99 | 96.27 | 0.6901 |
| gospel | 10 | 79.76 | 96.71 | 0.7359 |
| hip_hop | 10 | 89.92 | 98.62 | 0.4002 |
| rnb_soul | 10 | 84.99 | 97.06 | 0.5885 |

On this eval set F3 peaks on **hip_hop** (89.92%) and struggles most on **classical** (49.73%). The 11 per-genre [LoRA adapters](https://huggingface.co/PearlLeeStudio) are the recommended path for the genres beyond pop and jazz — F-series rows for the eight genres without a `[GENRE:X]` token in the 351-token vocabulary reflect decoding without genre conditioning (the token is omitted).

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy).
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences plus 2,500 pop rehearsal sequences (seed 42), pop:jazz ≈ 1.65:1. Sources across pretraining and fine-tuning: Chordonomicon (CC BY-NC 4.0), McGill Billboard (CC0), Jazz Harmony Treebank (public), JazzStandards / iReal Pro (community redistribution), Weimar Jazz Database (ODbL), JAAH (research-use). It resumed with the Phase-0 optimizer state and ran the applied F-series schedule given on the F1 card.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
- **Header tokens.** Small gaps between checkpoints can come from key and genre-marker predictions: epoch 9 predicts the placeholder C-major key that most test songs carry, which adds about 0.86 pop and 0.70 jazz points. Without the header targets the four rehearsal volumes from 1K to 10K lose pop within 0.11 points of each other at every epoch, so 2.5K is not a recovery point.

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
