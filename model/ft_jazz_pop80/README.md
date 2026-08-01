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

F1 slot of the pop→jazz mix-ratio sweep in *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026): the Phase-0 pop baseline fine-tuned on jazz with a 10,000-sequence pop rehearsal buffer, the pop-leaning endpoint and one of six released checkpoints. It is the base model the 11 per-genre `lora-*` adapters were trained on and pair with. Use it for chord-composition workflows targeting pop, rock, CCM, K-pop, J-pop, and modern country, and as the base for those adapters. F3 (`ft-pop50`) is the middle of the sweep, F4 (`ft-pop29`) the symmetric jazz-leaning endpoint, and [ft-pop80-v2](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2) the selection-corrected retrain of this slot.

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
| Best epoch | 6 (training-log, the released `best.pt` is the epoch-3 minimum-mixed-val selection, see Training data) |
| Released weights | Weight-identical to the [Phase-0 pop baseline](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline) by a full SHA-256 over the 25,841,152-element serialized state dict (25,661,440 unique parameters, with the tied input/output embedding stored as two tensors) |

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

The table gives the fine-tuning run's best-epoch (epoch 6) training-log metrics on the held-out pop test and the 6-source jazz test (167 sequences). The released weights do not embody them and score 72.86 jazz top-1 on the same test.

| Metric | Pop test | Jazz test (6-src) |
|---|---:|---:|
| Top-1 accuracy | **84.60%** | 81.03% |
| Top-5 accuracy | 96.96% | 92.41% |
| Perplexity | 1.78 | 2.31 |
| Δ vs. Phase-0 baseline | +0.39 | +8.17 |
| Δ vs. Phase-0, header targets excluded | −0.49 | +7.67 |

Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker). The last row leaves them out ([arXiv:2605.04998](https://arxiv.org/abs/2605.04998) v4, Table 6).

### Out-of-distribution per-genre baseline

F1 alone (no LoRA, released weights) on the 11 per-genre val splits the LoRA adapters target (10 Chordonomicon genre subsets plus the Bach-chorale classical split), with each sequence carrying its target genre marker. This is the no-LoRA reference reported on every `lora-<genre>` adapter card.

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

On this eval set the released F1 base peaks on **hip_hop** (90.66%) and struggles most on **classical** (49.55%). The 11 per-genre LoRA adapters ([sister `lora-*` repos](https://huggingface.co/PearlLeeStudio)) are the recommended path beyond pop and jazz. For the eight genres without a `[GENRE:X]` token in the 351-token vocabulary (all but rock, blues, and bossa) this eval gives the base no genre marker, and for blues and bossa it gives a marker the base never read in training.

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

All 1,513 jazz training sequences (Jazz Harmony Treebank, JazzStandards, Weimar Jazz Database, JAAH) plus 10,000 pop rehearsal sequences sub-sampled with seed 42 from the Phase-0 pop training split, pop:jazz ≈ 6.6:1 in the mix. The run resumed with the Phase-0 optimizer and scheduler state, which overrode the configured peak learning rate of 2 × 10⁻⁵ and two-epoch warmup (ten epochs maximum, patience 5): epoch 3 ran at the inherited learning rate 0 and updated no weights, and from epoch 4 the fine-tune's cosine schedule ran on the restored base rate of 3 × 10⁻⁴, giving 2.56 × 10⁻⁴, 2.07 × 10⁻⁴, 1.50 × 10⁻⁴, 9.26 × 10⁻⁵, and 4.39 × 10⁻⁵ for epochs 4 to 8 before early stopping. The training log's `lr` column holds each rate one row earlier, because a row is written after the end-of-epoch scheduler step. Best-checkpoint selection used minimum loss on a pop-dominated validation mix, which rose monotonically during jazz fine-tuning, so it kept the unchanged epoch 3.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
- **Header tokens.** Small gaps between checkpoints can come from key and genre-marker predictions: epoch 6 predicts the placeholder C-major key that most test songs carry, which adds about 0.86 pop and 0.70 jazz points, so the +0.39 pop delta is not a pop gain. Without the header targets the four rehearsal volumes from 1K to 10K lose pop within 0.11 points of each other at every epoch, the 10K run losing the least.
- **Same weights as Phase 0.** The released weights add no jazz adaptation or harmonic vocabulary over Phase 0, so the jazz-adapted F1 is ft-pop80-v2. Keeping the unchanged epoch depended on seed and data and was the worst case seen for this seed-42 run, since matched-data retrains with three seeds all selected adapted epochs (v2 card).
- **Genre markers it never read.** Of the out-of-distribution target markers only rock's occurs as an input in Phase-0 training, which maps Chordonomicon's tags onto pop, rock, and jazz, so the blues and bossa nova markers of the base vocabulary and the eight new rows copied from the equally unused `[GENRE:none]` row are new to it. Given `[GENRE:pop]` instead and scored without the header targets, F1 gains 1.0 to 1.9 pp on the test sets of the nine non-chorale genres other than rock and 3.6 pp on Bach chorales, and then comes within 0.12 pp of the released matched LoRA adapter in every non-chorale genre ([arXiv:2606.07334](https://arxiv.org/abs/2606.07334) v5), so outside Bach chorales the adapters mainly add a genre marker the model can use.

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
