# TheArtist Chord Generation Model Collection

PyTorch checkpoints and LoRA adapters released as companion artifacts to two papers: the seven base checkpoints accompany *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026; [arXiv:2605.04998](https://arxiv.org/abs/2605.04998)), and the eleven per-genre LoRA adapters accompany *How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?* (Lee, 2026; [arXiv:2606.07334](https://arxiv.org/abs/2606.07334)). The seven base checkpoints are 25.7M-parameter Music Transformers trained on a unified 351-token chord vocabulary spanning six pop and jazz corpora (six F-series runs plus the selection-corrected F1 retrain `ft_jazz_pop80_v2`). The eleven LoRA adapters extend the v1 F1 base toward additional genres.

This folder is the local staging area. Each subfolder maps one-to-one to a HuggingFace Hub repository under `PearlLeeStudio/`. Weight files (`best.pt`, `adapter_model.safetensors`) are gitignored and pushed directly to the Hub.

## Demo

[Watch TheArtist in action on YouTube](https://youtu.be/iSDj4VK9nKc) — interactive staff editor, MIDI input, AI generation with live SSE progress, voice-leading inversions, and per-genre LoRA playback across the 13-genre vocabulary.

## The seven base checkpoints (F-series + F1 v2)

| Folder | HF repo | Pop mix | Pop top-1 | Jazz top-1 | Role |
|---|---|---:|---:|---:|---|
| `phase0_pop_baseline/` | `PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline` | (pre-fine-tune) | **84.21** | 72.86 | Pop-dominated reference before jazz fine-tuning (5,609 of its 544,272 training songs carry the jazz marker) |
| `ft_jazz_pop80/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80` | 10,000 | **84.60** | 81.03 | F1, pop-leaning endpoint (paper run) |
| `ft_jazz_pop80_v2/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2` | 10,000 | 83.73 | 80.31 | F1 v2, selection-corrected (`select_best_on: jazz_val`), served as `ft_f1_v2` |
| `ft_jazz_pop67/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop67` | 5,000 | 84.06 | 79.91 | F2, 5K rehearsal |
| `ft_jazz_pop50/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50` | 2,500 | 84.20 | 80.99 | F3, middle of the sweep |
| `ft_jazz_pop29/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29` | 1,000 | 83.02 | **81.50** | F4, jazz-leaning endpoint |
| `ft_jazz_only/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-jazz-only` | 0 | 82.10 | 81.30 | F5, jazz-only forgetting reference |

The pop and jazz columns are each run's best-epoch training-log figures over all targets, jazz on the 6-source jazz test (167 sequences). They include the three header tokens that open every sequence (key, time signature, genre marker), which a checkpoint predicts almost the same way for every song: F4 and F5 predict the jazz marker even for pop songs (0.67 pop points lost, 1.07 jazz points gained), and the tabulated epochs of F1 and F3 predict the placeholder C-major key (about 0.86 pop and 0.70 jazz points). Scored without the header targets ([arXiv:2605.04998](https://arxiv.org/abs/2605.04998) v4, Table 6), every fine-tune gains 7.2 to 7.8 jazz points, F5 loses 1.48 pop points, and the four rehearsal volumes from 1K to 10K lose pop within 0.11 points of each other at every epoch, the 10K run losing the least. The smallest volume tested therefore already gives nearly all of the measured protection, and no volume stops a slow decline.

> **Released F1 weights.** The released `best.pt` of the v1 F1 (`ft_jazz_pop80` / `ft-pop80`) is weight-identical to the Phase-0 pop baseline, so its served numbers are Phase 0's (84.21 pop / 72.86 jazz). The run resumed from the Phase-0 checkpoint with its optimizer and scheduler state, whose learning rate had decayed to 0, so the first fine-tune epoch (epoch 3) updated no weights, and selection on minimum loss over the pop-dominated validation mix kept that unchanged epoch. The 84.60 / 81.03 above are the epoch-6 training-log figures, which were never released as `best.pt`. For three of the five fine-tunes (all but F2 and F5) the released minimum-mixed-validation checkpoint is not the tabulated epoch either. Keeping the unchanged epoch depended on seed and data: multi-seed retrains on the expanded 9-source corpus selected already-adapted epochs. The selection-corrected retrain `ft_jazz_pop80_v2` (`ft-pop80-v2`, served as `ft_f1_v2`, selected on a jazz-only validation subset) is hash-distinct from Phase 0 and scores 80.31 jazz on the same 6-source test. The 11 LoRA adapters below remain paired with the v1 F1 base.

## The eleven LoRA adapters

Per-genre LoRA adapters on the F1 base. Each genre uses the lowest-validation-loss snapshot of a 5-point rank sweep (r ∈ {4, 8, 16, 32, 64}, alpha = 2·r) on the attention Q/K/V projections. Training also updates the token-embedding and output matrices, so an adapter trains 465,920 (r = 4) to 1,940,480 (r = 64) parameters, 1.8 to 7.5% of the 25,849,344 that full fine-tuning updates. Adapter files are about 0.4 to 6 MB. Load them over the F1 base via `peft.PeftModel.from_pretrained`. License: **CC BY-NC 4.0** (inheriting Chordonomicon).

Each card reports top-1 over all targets, header tokens included. The frozen F1 base reads every target genre marker but rock's as an input it never saw in training, since Phase 0 maps Chordonomicon's tags onto pop, rock, and jazz. Scored without the header targets and with F1 given `[GENRE:pop]` instead ([arXiv:2606.07334](https://arxiv.org/abs/2606.07334) v5), the released matched adapters gain −0.02 to +0.12 pp in the ten Chordonomicon-derived genres, so there an adapter supplies a genre marker the model can use rather than better chord prediction. Only the Bach-chorale adapter adds clearly more (+9.6 pp). The same handicap explains why wrong-genre adapters often beat F1: they do so only where F1 reads an untrained marker, and against F1 given the pop marker only 4 of the 90 wrong-genre cells on the nine non-chorale sets other than rock score higher without the header targets, by at most 0.07 pp.

| Folder | HF repo | Genre | Training set |
|---|---|---|---:|
| `ft_f1_lora_country/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-country` | Country | ~62K |
| `ft_f1_lora_rock/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-rock` | Rock | ~49K |
| `ft_f1_lora_folk/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-folk` | Folk | ~61K |
| `ft_f1_lora_blues/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-blues` | Blues | ~10K |
| `ft_f1_lora_bossa/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-bossa` | Bossa nova | ~14K |
| `ft_f1_lora_electronic/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-electronic` | Electronic | ~15K |
| `ft_f1_lora_hip_hop/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-hip-hop` | Hip-Hop | ~14K |
| `ft_f1_lora_rnb_soul/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-rnb-soul` | R&B / Soul | ~10K |
| `ft_f1_lora_funk/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-funk` | Funk | ~3K |
| `ft_f1_lora_gospel/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-gospel` | Gospel | ~4K |
| `ft_f1_lora_classical/` | `PearlLeeStudio/TheArtist-MusicTransformer-lora-classical` | Classical | Bach chorales subset |

## What goes inside each subfolder before upload

**F-series base checkpoint folder:**
- `README.md` — HuggingFace model card (tracked in git)
- `config.json` — architecture metadata (tracked in git)
- `tokenizer.json` — 351-token chord vocabulary (tracked in git)
- `model.py`, `tokenizer.py` — copied from `ai/training/`
- `best.pt` — trained weights (gitignored, copied from `ai/checkpoints/<run>/best.pt`)

**LoRA adapter folder:**
- `README.md` — model card with PEFT front-matter (generated by `_generate_lora_cards.py`)
- `adapter_config.json`, `adapter_model.safetensors` — peft adapter (gitignored, copied from `ai/checkpoints/ft_f1_lora_<genre>/adapter/`)
- `embedding_extension.pt` — extra rows that grow the F1 vocab to cover new genre tokens (gitignored)
- `config.json`, `tokenizer.json`, `model.py`, `tokenizer.py` — base architecture + extended tokenizer

## Uploading to HuggingFace

One-time setup (uses the new `hf` CLI; the old `huggingface-cli` is deprecated as of huggingface_hub 1.0):

```bash
uv tool install huggingface_hub        # installs the `hf` CLI
hf auth login                          # paste a write-scope token from https://huggingface.co/settings/tokens
```

Per checkpoint, create the repo and upload the folder:

```bash
hf repo create PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50 --type model

hf upload \
  PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50 \
  ./model/ft_jazz_pop50 \
  . \
  --commit-message "Initial release"
```

Or use the helper scripts in this folder:

```bash
python model/_generate_lora_cards.py    # writes per-LoRA README.md from config.json
bash model/_stage_weights.sh            # stages all 18 folders (7 F-series + 11 LoRAs)
python model/_upload.py                 # uploads everything
python model/_upload.py --only-loras    # skip the F-series base checkpoints
python model/_upload.py --only-base     # skip the LoRA adapters
```

Token scope reminder: the token must have **write** access. Read-only tokens silently fail on `upload_folder`.

## Downloading for local serving

The backend (`backend/app/services/inference.py`) serves weights from
`ai/checkpoints/<run>/` — gitignored, so a fresh clone has none and the API runs
the rule-based fallback until they are present. `model/_download.py` is the
reverse of `_upload.py`: it pulls every public `PearlLeeStudio/*` repo (CC BY-NC,
no token) into the exact layout the loader expects.

```bash
uv run --with huggingface_hub python model/_download.py              # all 18 (7 F-series + 11 LoRA), ~2.2 GB
uv run --with huggingface_hub python model/_download.py --smoke      # just F1 (ft_jazz_pop80) — the serving default
uv run --with huggingface_hub python model/_download.py --only-base  # F-series only
uv run --with huggingface_hub python model/_download.py --only-loras # LoRA adapters only
```

Resulting layout (what `inference.CHECKPOINTS` + `_load_lora` expect):

```
ai/checkpoints/<run>/best.pt                                    # F-series, e.g. ft_jazz_pop80/best.pt
ai/checkpoints/ft_f1_lora_<genre>/adapter/adapter_config.json  # LoRA
ai/checkpoints/ft_f1_lora_<genre>/adapter/adapter_model.safetensors
ai/checkpoints/ft_f1_lora_<genre>/embedding_extension.pt
```

The Hub repos store the adapter files flat at the repo root; the downloader
re-nests them under `adapter/` (mirroring `ai/checkpoints/`) so
`PeftModel.from_pretrained` and the embedding overlay load without further moves.
The product registers exactly these 18 released selections (seven base models
and eleven adapters). Their availability depends on the local checkpoint files.

## Citation

Preprints: [arXiv:2605.04998](https://arxiv.org/abs/2605.04998) (base checkpoints) · [arXiv:2606.07334](https://arxiv.org/abs/2606.07334) (LoRA adapters).

```bibtex
@misc{lee2026chordmix,
  title         = {Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2605.04998},
  archivePrefix = {arXiv}
}

@misc{lee2026genreidentity,
  title         = {How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2606.07334},
  archivePrefix = {arXiv}
}
```
