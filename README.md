# TheArtist

An AI chord-composition tool for musicians. A custom Music Transformer (~25M parameters) generates chord progressions; the system serves the model behind an interactive staff editor with audio playback and MIDI input. Genre coverage is extended to 13 styles via per-genre LoRA adapters.

This repository contains the **public-facing summary**. The released artifacts are:

- **Paper** — [arXiv:2605.04998](https://arxiv.org/abs/2605.04998), *Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation* (Lee 2026)
- **Models** — [huggingface.co/PearlLeeStudio](https://huggingface.co/PearlLeeStudio) — Phase 0 baseline + F1–F5 fine-tunes + 11 per-genre LoRA adapters
- **Demo video** — *coming soon*

The application source code, training pipeline, evaluation scripts, and internal design notes are kept private. This document describes the project at the level of structure, datasets, and design choices.

---

## What it does

- **Compose on a staff.** VexFlow notation, click to select measures, drag-select ranges, slash-chord support.
- **AI continuations.** Three-suggestion generation pipeline; each suggestion ranked by physics-based (Sethares roughness) and theory-based reranking.
- **13-genre adaptation.** Frozen pop-jazz base model + 11 LoRA adapters covering blues, bossa, classical (Bach chorales), country, electronic, folk, funk, gospel, hip-hop, R&B/soul, rock; jazz and pop use the base directly.
- **Audio playback.** Tone.js with per-genre rule-based arrangement (harmony comping + bass + drum), 31-instrument override grid for the harmony layer.
- **MIDI input.** Hold-to-commit chord capture with voice-leading-aware confirmation flow.
- **Voice-leading inversions.** Slash-chord inversions injected post-generation so chord transitions sound smooth without the LM modeling inversions itself.

---

## Architecture (high level)

```
Browser (React 19 + Vite)        FastAPI :8000              AI Pipeline (PyTorch)
─────────────────────────       ─────────────              ─────────────────────────
Interactive staff (VexFlow)  →  /api/generate         →    Music Transformer (~25M)
Audio + MIDI playback           per-genre LoRA routing     + Sethares roughness rerank
Voicing visualization           rule-based arrangement     + theory-corpus rerank (RAG)
                                voice-leading injection
```

Three modules: `frontend/` (React + Vite, staff editor + audio playback), `backend/` (FastAPI, model serving + reranking + voice leading), `ai/` (Music Transformer training, LoRA library, physics + RAG rerank).

---

## Project structure (top level)

```
TheArtist/
├── frontend/        React 19 + TypeScript + Vite, VexFlow + Tone.js
├── backend/         FastAPI + Pydantic, Python 3.12+ (uv)
├── ai/
│   ├── training/    Music Transformer + LoRA training pipeline
│   ├── physics/     Sethares roughness rerank (R1)
│   ├── rag/         Theory-corpus retrieval rerank (R2)
│   └── checkpoints/ Released to HuggingFace (link above)
├── paper/           arXiv submission bundle
└── model/           HuggingFace model card sources
```

---

## Datasets

### Training corpora

| Dataset | Songs (used) | Genre | License |
|---|---:|---|---|
| [Chordonomicon](https://huggingface.co/datasets/ailsntua/Chordonomicon) | ~679K | Pop / rock + community-tagged subsets (country, alt, rap, soul, electronic) | CC BY-NC 4.0 |
| [McGill Billboard](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) | ~890 | Pop / rock | CC0 |
| [Jazz Harmony Treebank (JHT)](https://github.com/DCMLab/JazzHarmonyTreebank) | ~1,170 | Jazz | Public |
| [Weimar Jazz Database (WJazzD)](https://jazzomat.hfm-weimar.de/) | ~283 | Jazz | ODbL |
| [JAAH](https://mtg.github.io/JAAH/) | ~113 | Jazz | Research |
| [JazzStandards / iReal-derived](https://github.com/mikeoliphant/JazzStandards) | ~293 | Jazz | Community corpus |
| [Bach 371 chorales (via music21)](https://web.mit.edu/music21/) | 371 | Tonal chorales (used as the "classical" LoRA target) | Public domain |

Raw datasets are **not redistributed** in this repo. Chordonomicon is CC BY-NC; the others have their own per-source terms. Acquire each from the upstream link above.

For the LoRA expansion (R4), the 11 target genres draw from Chordonomicon's genre-labeled subsets; see paper §4 for selection-bias notes (e.g., the "electronic" slice is biased toward synthwave / melodic house, "hip-hop" toward jazz-rap / neo-soul).

### Evaluation

- **Per-genre validation splits.** 80/10/10 song-level split of each training corpus. Token-level top-1 / top-5 chord prediction with teacher forcing. Used for the rank sweep (paper §7) and per-genre baseline reporting.
- **Real-song evaluation set.** 130 curated songs (13 genres × 10 songs), drawn only from validation/test splits of the corpora above (no train leakage). Used for HuggingFace model-card sanity reporting.

### Theory retrieval (R2 RAG)

The reranking layer indexes two corpora:

- **Berklee Book of Jazz Harmony** — copyrighted, **not redistributed**. Acquire your own copy.
- **Open Music Theory v2** — [open license](https://viva.pressbooks.pub/openmusictheory/), can be re-ingested from the upstream source.

The PDF text extraction, paragraph chunking, embedding (Snowflake `arctic-embed-l-v2`), and topic tagging (Gemma 3 12B via ollama) pipeline lives in `ai/rag/` (private).

---

## Not in this repository

- Frontend / backend / training source code — private. Demo video covers the end-user experience.
- Trained checkpoints — published on HuggingFace (link above).
- Raw datasets — license + size constraints; see Datasets section.
- Berklee Book of Jazz Harmony PDF — copyrighted.
- Internal design notes, deliberation logs, and roadmap drafts.

---

## Citation

```bibtex
@misc{lee2026chordmix,
  title         = {Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2605.04998},
  archivePrefix = {arXiv}
}
```

A second draft on chord-level LoRA adapters across 11 genres is in preparation (arXiv v2 / ISMIR submission).

---

## License

Copyright (c) 2026 PearlLee Studio. All Rights Reserved.

Public release scope: this README, the paper at arXiv, and the model checkpoints on HuggingFace. All other source code and data are private.

The released checkpoints are derived from CC BY-NC 4.0 training data (Chordonomicon) and are therefore distributed for non-commercial paper / portfolio / demo use only.

Contact: pearl1379@gmail.com
