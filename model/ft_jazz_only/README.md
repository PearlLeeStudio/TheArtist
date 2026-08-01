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
---

# TheArtist Music Transformer — F5 (Jazz Only, no pop rehearsal)

Jazz-only fine-tune of the Phase-0 pop baseline with **no pop rehearsal data** — the catastrophic-forgetting endpoint of the F1–F5 mix-ratio series from *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026), evaluated further in [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334). The model generates chord-symbol progressions from a key- and genre-conditioned prompt. F5 is strictly dominated by F4 on every axis and should not be selected as a production checkpoint; it is released for replication of the per-epoch forgetting curve and for direct inspection of the failure mode.

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
| Best epoch | 7 |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-jazz-only")
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

For per-genre adaptation beyond pop and jazz, see the 11 LoRA adapter repos at [PearlLeeStudio](https://huggingface.co/PearlLeeStudio) — they chain on top of the released `ft-pop80` (F1) base, not this checkpoint.

## Evaluation

Teacher-forced token-level metrics on the held-out pop and jazz test sets:

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | 82.10% | 81.30% |
| Top-5 accuracy | 96.31% | 92.44% |
| Perplexity | 1.96 | 2.24 |
| Δ vs. Phase-0 baseline | −2.11 | +8.44 |

F5 illustrates the catastrophic-forgetting failure mode that motivated the paper: pop top-1 collapses by 2.11 points — the steepest drop (1.37 points) coming in the first epoch — and does not recover, while jazz top-1 reaches 81.30% against F4's 81.50% (F4 also keeps an extra 0.92 points of pop).

Known failure modes of this checkpoint specifically:

- Chord progressions trend toward dense chromatic voicings that are commercially niche.
- Generations on pop prompts retain diatonic structure but show persistent chromatic substitution.
- See the paper's §6.4 and §7.5 for representative continuations.

<!-- real-song-eval:begin -->
### Per-genre real-song eval

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 85.97 | 95.78 | 0.6333 |
| rock | 10 | 85.38 | 96.46 | 0.5710 |
| jazz | 10 | 71.16 | 86.77 | 1.2883 |
| blues | 10 | 79.84 | 92.63 | 0.9230 |
| bossa | 10 | 81.01 | 95.13 | 0.8129 |
| classical | 10 | 46.86 | 77.96 | 2.3232 |
| country | 10 | 84.81 | 97.04 | 0.6185 |
| electronic | 10 | 86.49 | 97.59 | 0.5834 |
| folk | 10 | 83.70 | 98.07 | 0.6290 |
| funk | 10 | 82.42 | 94.96 | 0.7809 |
| gospel | 10 | 78.40 | 95.58 | 0.8520 |
| hip_hop | 10 | 89.99 | 98.12 | 0.4733 |
| rnb_soul | 10 | 84.20 | 96.47 | 0.6757 |

On this eval set F5 peaks on **hip_hop** (89.99%) and struggles most on **classical** (46.86%). On the eight genres whose `[GENRE:X]` tokens do not exist in the 351-token vocabulary (all beyond pop/jazz except rock, blues, and bossa) the F-series decodes without genre conditioning, so the per-genre [LoRA adapters](https://huggingface.co/PearlLeeStudio) are the recommended path beyond pop and jazz.

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy).
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences; no pop rehearsal data. Fine-tuned from the Phase-0 pop baseline, whose pretraining corpus is dominated by the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon) (CC BY-NC 4.0).

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
