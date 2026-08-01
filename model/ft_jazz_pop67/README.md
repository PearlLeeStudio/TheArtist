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

# TheArtist Music Transformer — F2 (Pop 5K Mix)

F2 in the pop→jazz fine-tuning sweep: the Phase-0 pop baseline fine-tuned on all 1,513 jazz training sequences with a 5,000-sequence pop rehearsal buffer. It sits between the pop-leaning F1 and the sweet-spot F3; the mix-ratio paper finds it dominated by F3 on every axis, so it is released for reproducibility of the saturation curve rather than as a recommended operating point (prefer F3 for the balanced setting, F1 for pop-leaning, F4 for jazz-leaning). One of six checkpoints from *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026).

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

## Model details

| Field | Value |
|---|---|
| Architecture | Music Transformer with relative positional attention |
| Parameters | 25,661,440 |
| Vocabulary size | 351 tokens |
| Max sequence length | 256 |
| d_model / heads / FFN / layers | 512 / 8 / 2048 / 8 |
| Fine-tune resumed from | Phase-0 pop baseline |
| Fine-tune mix | 1,513 jazz + 5,000 pop rehearsal (pop:jazz ≈ 3.3:1, seed 42) |
| Best epoch | 4 |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop67")
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

For per-genre adaptation beyond pop and jazz, see the 11 [LoRA adapter repos](https://huggingface.co/PearlLeeStudio) — they chain on top of the released `ft-pop80` (F1) base, not this checkpoint.

## Evaluation

Held-out per-genre test sets:

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | 84.06% | 79.91% |
| Top-5 accuracy | 96.17% | 91.46% |
| Perplexity | 1.75 | 2.34 |
| Δ vs. Phase-0 baseline | −0.15 | +7.05 |

<!-- real-song-eval:begin -->
### Per-genre real-song eval

Teacher-forced token-level metrics per genre; on this set F2 peaks on **hip_hop** (90.49%) and struggles most on **classical** (49.84%). The 11 per-genre [LoRA adapters](https://huggingface.co/PearlLeeStudio) remain the recommended path beyond pop and jazz — for the eight genres without a `[GENRE:X]` token in the 351-token vocabulary (all but rock, blues, and bossa) the F-series numbers reflect decoding without genre conditioning.

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 86.23 | 95.70 | 0.5810 |
| rock | 10 | 87.04 | 96.90 | 0.4660 |
| jazz | 10 | 69.43 | 85.73 | 1.3464 |
| blues | 10 | 83.11 | 93.35 | 0.7935 |
| bossa | 10 | 82.44 | 95.04 | 0.7304 |
| classical | 10 | 49.84 | 83.17 | 2.0839 |
| country | 10 | 85.80 | 97.25 | 0.5200 |
| electronic | 10 | 86.81 | 97.49 | 0.5097 |
| folk | 10 | 84.60 | 98.06 | 0.5295 |
| funk | 10 | 83.76 | 95.61 | 0.6983 |
| gospel | 10 | 80.47 | 95.83 | 0.7463 |
| hip_hop | 10 | 90.49 | 97.62 | 0.3990 |
| rnb_soul | 10 | 84.66 | 96.40 | 0.5954 |

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy).
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences plus 5,000 pop rehearsal sequences (seed 42). Pop:jazz ≈ 3.3:1. Sources across pretraining and fine-tuning: Chordonomicon (CC BY-NC 4.0), McGill Billboard (CC0), Jazz Harmony Treebank (public), JazzStandards (community redistribution), Weimar Jazz Database (ODbL), JAAH (research-use).

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
