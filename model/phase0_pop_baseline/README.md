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
- pop
---

# TheArtist Music Transformer — Phase-0 (Pop Baseline)

Pop-only pretraining baseline of the mix-ratio study — a Music Transformer trained from scratch on ~544K pop songs × 3 epochs (Chordonomicon + McGill Billboard). All five F-series jazz fine-tunes resume from this checkpoint, making it the no-jazz reference for measuring catastrophic forgetting in the paper [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026).

It is the most pop-fluent and most jazz-naive model in the collection — use it when pop output is the only target; for balanced pop/jazz output use F3 ([ft-pop50](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50)), for jazz-leaning output F4 ([ft-pop29](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29)).

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

## Model details

| Field | Value |
|---|---|
| Architecture | Music Transformer with relative positional attention |
| Parameters | 25,661,440 |
| Vocabulary size | 351 tokens |
| Max sequence length | 256 |
| d_model / heads / FFN / layers | 512 / 8 / 2048 / 8 |
| Training framework | PyTorch 2.5+ (CUDA 12.1) |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline")
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

For per-genre adaptation beyond pop and jazz, see the 11 LoRA adapter repos at [PearlLeeStudio](https://huggingface.co/PearlLeeStudio).

## Evaluation

Held-out per-genre test sets:

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | **84.21%** | 72.86% |
| Top-5 accuracy | 97.09% | 86.51% |
| Perplexity | 1.73 | 4.01 |

The 72.86% jazz top-1 from a pop-only model reflects the substantial token overlap between the two genres; jazz-specific gains in the fine-tuned checkpoints come from learning the transition statistics over those shared tokens, not from learning new tokens.

<!-- real-song-eval:begin -->
### Per-genre real-song eval

Teacher-forced next-token top-1 / top-5 / cross-entropy over each song's full token sequence (truncated to `max_seq_len=256`). On this set Phase-0 peaks on **hip_hop** (90.66% top-1) and struggles most on **classical** (49.55%).

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

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy). Source license summary: McGill Billboard (CC0), Jazz Harmony Treebank / JazzStandards / WJazzD (Public), Bach chorales (Public Domain), Chordonomicon per-genre subsets (CC BY-NC 4.0).
<!-- real-song-eval:end -->

## Training data

Trained from scratch on the pop training split — ~544K songs drawn from the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon) (CC BY-NC 4.0) and McGill Billboard (CC0), lightly filtered and twelve-key-augmented. Three epochs at peak learning rate 3 × 10⁻⁴ with one-epoch warmup and cosine decay; wall-clock time ≈27 hours on a single NVIDIA GeForce RTX 4070 Laptop GPU.

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
