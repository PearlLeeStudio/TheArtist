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

# TheArtist Music Transformer — F1 v2 (Pop 10K Mix, selection-corrected retrain)

Selection-corrected retrain of the F1 slot: the Phase-0 pop baseline fine-tuned on jazz with a 10,000-sequence pop rehearsal buffer, identical in recipe to [ft-pop80](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80) except that the best checkpoint is selected on a jazz-only validation subset (`select_best_on: jazz_val`) instead of the pop-dominated validation mix. Use it for pop and rock with genuine jazz coloration, where jazz substitutions (secondary dominants, ii-V detours) appear selectively over a pop harmonic root, and choose [ft-pop29 (F4)](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29) when jazz is the primary target. It is a standalone base: the 11 per-genre `lora-*` adapters were trained on the v1 base and pair with it, not with this checkpoint. One of the models from [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026).

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
| Best epoch | 6 (selected by jazz-only val loss, 0.9331) |
| Trained on | Google Colab Tesla T4, AMP, ~20 min/epoch |
| Released weights | Hash-distinct from Phase 0 (full-state SHA-256 `b47144d4db058638…` vs `3875f40784f19811…`), unlike the v1 release, which is weight-identical to Phase 0 |

## Usage

Requires `torch`, `huggingface_hub`. The repo bundles `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2")
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

## Evaluation

Teacher-forced metrics at the selected epoch 6 on the held-out pop test and the 6-source jazz test, against the un-adapted initialization:

| Metric | Pop test | Jazz test (6-src, 167 sequences) |
|---|---:|---:|
| Top-1 accuracy | 83.73% | **80.31%** |
| Top-5 accuracy | 96.95% | 92.42% |
| Perplexity | 1.78 | 2.31 |
| Δ vs. un-adapted init | −0.48 | **+7.45** |

Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker). On the unified 9-source jazz test (501 sequences) the same weights score **75.05**, so figures across the two test sets are not cell-comparable. In matched-data retrains on the 9-source corpus (seeds 42, 123, 7), jazz-val selection gives jazz top-1 75.76 ± 0.03, all hash-distinct from Phase 0. The full per-epoch training curve, including the `jazz_val_*` selection-metric columns, ships in this repo's `eval_results.csv`.

<!-- real-song-eval:begin -->
### Per-genre real-song eval

Paired against Phase-0 on the same songs, this checkpoint gains **+5.17 pp** mean jazz top-1 (median +5.19, 6/10 songs win) at −0.42 pp pop, with all other genres within ±1.1 pp. For the eight genres without a `[GENRE:X]` token in the 351-token vocabulary (all beyond pop/jazz except rock, blues, and bossa) the model runs without a genre tag — the per-genre [`lora-*` adapters](https://huggingface.co/PearlLeeStudio), paired with the v1 base, are the recommended path there. 130 songs (10 per genre × 13 genres, seed 42) drawn from held-out val/test partitions — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0; titles are Spotify track IDs by upstream policy).

| Genre | n_songs | Top-1 (%) | Top-5 (%) | val_loss |
|---|---:|---:|---:|---:|
| pop | 10 | 86.26 | 95.74 | 0.6167 |
| rock | 10 | 86.59 | 97.32 | 0.4877 |
| jazz | 10 | 70.13 | 86.42 | 1.3382 |
| blues | 10 | 82.61 | 94.26 | 0.7972 |
| bossa | 10 | 82.02 | 96.00 | 0.7641 |
| classical | 10 | 49.27 | 81.63 | 2.1650 |
| country | 10 | 85.62 | 98.04 | 0.5327 |
| electronic | 10 | 86.85 | 98.50 | 0.5087 |
| folk | 10 | 84.58 | 98.62 | 0.5525 |
| funk | 10 | 83.12 | 95.79 | 0.7053 |
| gospel | 10 | 80.10 | 96.76 | 0.7609 |
| hip_hop | 10 | 90.09 | 98.40 | 0.3965 |
| rnb_soul | 10 | 84.95 | 96.74 | 0.6152 |
<!-- real-song-eval:end -->

## Training data

All 1,513 jazz training sequences (Jazz Harmony Treebank, JazzStandards, Weimar Jazz Database, JAAH) plus 10,000 pop rehearsal sequences sub-sampled with seed 42 from the Phase-0 pop training split, the same mix as v1, drawn from the rebuilt 6-source splits. The configuration is identical to v1 (peak learning rate 2 × 10⁻⁵, two-epoch warmup, ten epochs maximum with patience 5) except `select_best_on: jazz_val`, and as in v1 the restored Phase-0 optimizer and scheduler state overrode the rate and warmup: epoch 3 ran at the inherited learning rate 0, so `epoch_0003.pt` is bit-identical to Phase 0, and epochs 4 to 9 followed a cosine on the restored base rate of 3 × 10⁻⁴ at 2.56 × 10⁻⁴, 2.07 × 10⁻⁴, 1.50 × 10⁻⁴, 9.26 × 10⁻⁵, 4.39 × 10⁻⁵, and 1.14 × 10⁻⁵ (each logged one row earlier in the `lr` column). The mixed validation loss rose monotonically (0.5578 → 0.5929), so min-mixed-val selection would have kept the un-adapted epoch 3, while jazz-only selection picks epoch 6.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
- **Header tokens.** Small gaps between checkpoints can come from key and genre-marker predictions: an epoch that predicts the placeholder C-major key that most test songs carry gains about 0.86 pop and 0.70 jazz points ([arXiv:2605.04998](https://arxiv.org/abs/2605.04998) v4, Section 5.4). This card reports all-target figures only.

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
