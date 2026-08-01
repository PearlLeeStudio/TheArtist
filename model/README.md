# TheArtist Chord Generation Model Collection

PyTorch checkpoints and LoRA adapters released as companion artifacts to two papers: the seven base checkpoints accompany *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee, 2026; [arXiv:2605.04998](https://arxiv.org/abs/2605.04998)), and the eleven per-genre LoRA adapters accompany *How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?* (Lee, 2026; [arXiv:2606.07334](https://arxiv.org/abs/2606.07334)). The seven base checkpoints are 25M-parameter Music Transformers trained on a unified 351-token chord vocabulary spanning six pop and jazz corpora (six F-series runs plus the selection-corrected F1 retrain `ft_jazz_pop80_v2`); the eleven LoRA adapters extend the v1 F1 base toward additional genres.

This folder is the local staging area. Each subfolder maps one-to-one to a HuggingFace Hub repository under `PearlLeeStudio/`. Weight files (`best.pt`, `adapter_model.safetensors`) are gitignored and pushed directly to the Hub.

## Demo

[Watch TheArtist in action on YouTube](https://youtu.be/iSDj4VK9nKc) — interactive staff editor, MIDI input, AI generation with live SSE progress, voice-leading inversions, and per-genre LoRA playback across the 13-genre vocabulary.

## The seven base checkpoints (F-series + F1 v2)

| Folder | HF repo | Pop mix | Pop top-1 | Jazz top-1 | Role |
|---|---|---:|---:|---:|---|
| `phase0_pop_baseline/` | `PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline` | (pre-fine-tune) | **84.21** | 72.86 | Pop reference, no jazz exposure |
| `ft_jazz_pop80/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80` | 10,000 | **84.60** | 81.03 | F1, pop-leaning endpoint (paper run) |
| `ft_jazz_pop80_v2/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2` | 10,000 | 83.73 | 80.31 | F1 v2, selection-corrected (`select_best_on: jazz_val`), served as `ft_f1_v2` |
| `ft_jazz_pop67/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop67` | 5,000 | 84.06 | 79.91 | F2, dominated calibration |
| `ft_jazz_pop50/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50` | 2,500 | 84.20 | 80.99 | **F3, balanced sweet spot** |
| `ft_jazz_pop29/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29` | 1,000 | 83.02 | **81.50** | F4, jazz-leaning endpoint |
| `ft_jazz_only/` | `PearlLeeStudio/TheArtist-MusicTransformer-ft-jazz-only` | 0 | 82.10 | 81.30 | F5, jazz-only forgetting reference |

> **F1 weight-identity erratum (2026-06-11).** The v1 F1 (`ft_jazz_pop80` / `ft-pop80`) released `best.pt` is weight-identical to the Phase-0 pop baseline — a best-checkpoint-selection artifact (min loss on the pop-dominated val mix retained the un-adapted init), so its served numbers match Phase 0 (84.21 pop / 72.86 jazz; the 84.60 / 81.03 above are the paper's epoch-6 training-log figures, never released as `best.pt`). **The F-series jazz top-1 figures in the table above (81.03, 79.91, 80.99, 81.50, 81.30) are best-epoch training metrics on the 6-source jazz test (167 sequences), not the released checkpoints'** — for 3/5 runs (only F2/F5 match) the min-mixed-val checkpoint differs from the best jazz epoch. The selection-corrected retrain `ft_jazz_pop80_v2` (`ft-pop80-v2`, served as `ft_f1_v2`) is hash-distinct from Phase 0 and embodies the intended jazz adaptation (80.31 jazz on the same 6-source test). *(2026-06-13: the "min-mixed-val retained the un-adapted init" outcome is **seed/data-dependent** — it occurred for the seed-42 6-source run because min-mixed-val landed on the warmup epoch; multi-seed retrains on the expanded 9-source corpus selected already-adapted epochs. The robust claim is that jazz_val selection reliably yields an adapted base across seeds.)* The 11 LoRA adapters below remain paired with the v1 F1 base.

## The eleven LoRA adapters

Per-genre LoRA adapters on the F1 base. Each genre uses its best-rank snapshot from a 5-point sweep (r ∈ {4, 8, 16, 32, 64}; alpha = 2·r), applied to attention Q/K/V projections, ~106K–1.58M trainable parameters (~0.4%–6.1% of base). Adapter files are ~0.4–6 MB; load over the F1 base via `peft.PeftModel.from_pretrained`. License: **CC BY-NC 4.0** (inheriting Chordonomicon).

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
One internal selector key has no Hub source and stays unavailable
unless trained locally.

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
