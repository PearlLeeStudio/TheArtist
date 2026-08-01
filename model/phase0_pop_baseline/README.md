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

Pop-dominated pretraining baseline of the mix-ratio study, trained from scratch. All five F-series jazz fine-tunes resume from it, making it the pre-fine-tune reference for catastrophic forgetting in *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026), and since the released F1 is weight-identical to it, it is in effect the frozen base of [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026). Use it when pop is the only target. For pop and jazz the choice among the rehearsal fine-tunes is stylistic: F3 ([ft-pop50](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50)) sits at an intermediate register and F4 ([ft-pop29](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29)) commits more to jazz.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Web app](https://artist.pearlleestudio.co) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

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

Held-out pop and jazz test sets:

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | **84.21%** | 72.86% |
| Top-5 accuracy | 97.09% | 86.51% |
| Perplexity | 1.73 | 4.01 |
| Top-1, header targets excluded | 84.87% | 74.22% |

Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker). The last row leaves them out ([arXiv:2605.04998](https://arxiv.org/abs/2605.04998) v4, Section 5.4).

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

The pop training split holds ~544K songs from the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon) (CC BY-NC 4.0) and McGill Billboard (CC0), lightly filtered and twelve-key-augmented. Chordonomicon's genre tags are mapped onto three markers, so of the 544,272 training songs 425,274 carry `[GENRE:pop]`, 113,389 `[GENRE:rock]`, and 5,609 `[GENRE:jazz]`. Three epochs ran under a per-epoch cosine (epochs 0 and 1 at the peak 3 × 10⁻⁴, epoch 2 at 1.5 × 10⁻⁴), so the configured one-epoch warmup had no effect, in ≈27 hours on one NVIDIA GeForce RTX 4070 Laptop GPU.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
- **Header and bar tokens.** Top-1 moves with the key and genre-marker predictions (see the header-free row), and bar tokens, which this model predicts 95% of the time, supply 39 of its roughly 73 jazz points. On jazz chord targets alone it is right 60% of the time, mostly on diatonic continuations both genres share, and chord targets hold 22 of the 27 points it misses.

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
