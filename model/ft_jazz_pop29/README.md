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

# TheArtist Music Transformer — F4 (Pop 1K Mix, jazz-leaning)

Jazz-adapted chord model fine-tuned from the Phase-0 pop baseline with a 1,000-sequence pop rehearsal buffer — the jazz-leaning endpoint of the mix-ratio sweep, one of six checkpoints released with *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026). It posts the highest jazz top-1 accuracy in the collection (81.50%) at the cost of 1.19 pop points, and is TheArtist app's recommended checkpoint for jazz. Recommended for jazz-flavoured chord composition (chord symbols only — no melody or audio generation); F3 (`ft-pop50`) is the balanced alternative and F1 (`ft-pop80`) the pop-leaning endpoint.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

> **Served-weights note.** The released `best.pt` is the minimum-mixed-validation-loss checkpoint (epoch 4 — pop 83.38 / jazz 80.68 on the same test sets); the Evaluation table below shows the paper's epoch-6 best-jazz figures. The released F1, cited here as the pop-leaning endpoint, is itself weight-identical to the Phase-0 baseline (a checkpoint-selection artifact — see the F1 card); this checkpoint's weights are hash-distinct from both.

## Model details

| Field | Value |
|---|---|
| Architecture | Music Transformer with relative positional attention |
| Parameters | 25,661,440 |
| Vocabulary size | 351 tokens |
| Max sequence length | 256 |
| d_model / heads / FFN / layers | 512 / 8 / 2048 / 8 |
| Fine-tune resumed from | Phase-0 pop baseline |
| Best epoch | 6 (released `best.pt` = epoch 4; see note above) |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29")
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

Held-out per-genre test sets:

| Metric | Pop test | Jazz test |
|---|---:|---:|
| Top-1 accuracy | 83.02% | **81.50%** |
| Top-5 accuracy | 96.93% | 92.59% |
| Perplexity | 1.80 | 2.26 |
| Δ vs. Phase-0 baseline | −1.19 | **+8.64** |

F4 produces the most jazz-flavoured continuations among the released checkpoints — secondary dominants, tritone substitutions, modal interchange, and ii-V chains across distant keys — at the cost of roughly one point of pop top-1 accuracy.

<!-- real-song-eval:begin -->
### Per-genre real-song eval

Teacher-forced token-level metrics; on this set F4 peaks on **hip_hop** (90.34%) and struggles most on **classical** (49.65%). This is auxiliary signal — the 11 per-genre [LoRA adapters](https://huggingface.co/PearlLeeStudio) are the recommended path beyond pop and jazz — for the eight genres without a matching token in the 351-token vocabulary (all but rock, blues, and bossa) the F-series decodes without genre conditioning.

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 85.57 | 95.91 | 0.5859 |
| rock | 10 | 87.28 | 97.50 | 0.4677 |
| jazz | 10 | 71.42 | 85.54 | 1.3367 |
| blues | 10 | 81.99 | 93.86 | 0.7970 |
| bossa | 10 | 82.62 | 95.73 | 0.7226 |
| classical | 10 | 49.65 | 81.51 | 2.1079 |
| country | 10 | 86.30 | 98.18 | 0.5191 |
| electronic | 10 | 86.81 | 98.48 | 0.5100 |
| folk | 10 | 84.81 | 98.63 | 0.5337 |
| funk | 10 | 83.39 | 96.13 | 0.6959 |
| gospel | 10 | 80.25 | 96.58 | 0.7343 |
| hip_hop | 10 | 90.34 | 98.58 | 0.3982 |
| rnb_soul | 10 | 85.04 | 97.04 | 0.5907 |

130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy).
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences plus 1,000 pop rehearsal sequences (seed 42). Pop:jazz ≈ 0.66:1 — less pop than jazz in the mix. Underlying corpora and licenses: Chordonomicon (CC BY-NC 4.0), McGill Billboard (CC0), Jazz Harmony Treebank (public), JazzStandards / iReal Pro (community redistribution), Weimar Jazz Database (ODbL), JAAH (research-use public).

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
