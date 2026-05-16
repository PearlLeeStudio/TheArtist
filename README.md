<h1 align="center">TheArtist</h1>

<p align="center">
  <em>An AI chord-composition tool for musicians — write progressions on an interactive staff, get suggestions from a custom-trained Music Transformer, and see voice-leading visualised in real time.</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.04998"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.04998-b31b1b?style=flat-square"></a>
  <a href="https://huggingface.co/PearlLeeStudio"><img alt="HuggingFace" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-PearlLeeStudio-ffd21e?style=flat-square&labelColor=1c1917"></a>
  <img alt="License" src="https://img.shields.io/badge/license-portfolio-737373?style=flat-square">
  <img alt="Stack" src="https://img.shields.io/badge/stack-React%2019%20%C2%B7%20FastAPI%20%C2%B7%20PyTorch-1c1917?style=flat-square">
</p>

<p align="center">
  <img src="screenshot/01-main-light.png" alt="TheArtist — main view with chord progression and voicing keyboard (light theme)" width="100%">
</p>

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

## Walkthrough

### AI generation with live progress

<img src="screenshot/03-generate.png" alt="Generate tab — model selector, slot selected, three suggestions returned">

Click **Generate** with a slot range selected. The right panel streams pipeline phases via SSE — *Compose 1/3 → 2/3 → 3/3 → Rank → Voice → Done* — so the 5–10 s inference window narrates itself. Three suggestions land, ranked by Sethares roughness + theory-corpus retrieval composite score.

### Voicing — piano + guitar

<img src="screenshot/04-voicing.png" alt="Right panel with voicing keyboard and guitar fretboard">

Whichever chord is active (selected on the staff, currently playing back, or under a MIDI hold) lights up its tones on a styled keyboard or fretboard. Voicings cap at 4 notes; switch instruments live mid-playback.

### MIDI input

<img src="screenshot/06-midi.png" alt="MIDI input panel — Enable MIDI button, hold-to-commit hint">

Pick a slot on the staff, hit **Enable MIDI** with a connected keyboard. Hold a chord ≥3 s and it auto-commits. Release early and you get a Confirm card with root-cycling chips (`C6` ↔ `Am7/C` etc.) plus inversion detection.

### Dark + light themes

<table>
<tr>
<td width="50%" valign="top"><img src="screenshot/01-main-light.png" alt="Light theme"></td>
<td width="50%" valign="top"><img src="screenshot/02-main-dark.png" alt="Dark theme"></td>
</tr>
<tr>
<td><sub><strong>Light</strong> — pure-achromatic R=G=B ramp on Tailwind's neutral axis.</sub></td>
<td><sub><strong>Dark</strong> — near-black editorial palette (#0a0a0a base).</sub></td>
</tr>
</table>

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

## Model serving for upstream agents

`POST /api/generate/song` is the integration point for upstream creative-AI agents (e.g. TheVoyager). The caller sends a genre and length; TheArtist owns key / BPM / time signature / chord-generation checkpoint / per-track instrument selection internally and returns a full 3-track render.

```bash
curl -sS http://localhost:8000/api/generate/song \
  -H 'Content-Type: application/json' \
  -d '{"genre": "jazz", "length_bars": 8}'
```

**Response shape:** `{key, bpm, time_signature, bars, tracks: {harmony, bass, drum}, midi_b64, model_used}`. Each track exposes per-event `{bar, beat, pitch, duration, velocity}` plus a GM Soundfont instrument name. Drum events use voice letters (`K`/`S`/`H`/...) mapped to GM channel-10 percussion. `midi_b64` is a base64-encoded MIDI render bundling the three tracks for the caller's downstream synthesis.

**Genre routing.** The chord layer dispatches to the appropriate checkpoint (jazz → F4, pop → F1, others → matching LoRA). Harmony, bass, and drum are deterministic per-genre patterns in v1; learned chord-conditioned multitrack arrangement is the next research step.

Supported genres: `jazz, pop, rock, blues, bossa, classical, country, rnb_soul, hip_hop, electronic, funk, folk, gospel`. `length_bars` accepts 1–64.

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

For the LoRA expansion, the 11 target genres draw from Chordonomicon's genre-labeled subsets; see paper §4 for selection-bias notes (e.g., the "electronic" slice is biased toward synthwave / melodic house, "hip-hop" toward jazz-rap / neo-soul).

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
