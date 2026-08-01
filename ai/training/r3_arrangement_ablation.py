"""§3.2 R3 ablation — does swapping F1 → F1+LoRA at the chord layer
produce measurably different arrangement renders downstream?

The deployed system renders multi-track output as:

    chord stream  ──►  song_builder.build_*  ──►  {harmony, bass, drum} events

The arrangement layer (bass + drum + comping rhythm) is purely rule-based and
pulled from per-genre tables in `backend/app/services/genre_tables.py`
(`GENRE_HARMONY_RHYTHM` / `GENRE_BASS_RHYTHM` / `GENRE_DRUM_PATTERN`). So any
difference between F1 vs F1+LoRA arrangement output is driven *entirely* by
the chord sequence the model proposes — every other knob is a genre invariant.

Token-level chord top-1 (the §3.1/§3.3 metric) does not apply here because
arrangement adds bass/drum events on a different axis. This script defines
a small set of arrangement-side metrics that are sensitive to chord-sequence
variation; the metrics that are pure rule-of-genre invariants
(e.g. `bass_event_count` for a fixed rhythm hitting different roots) are
reported only for completeness.

Per-prompt protocol (per genre):
  1. Pull the genre's GENRE_DEFAULTS for (key, bpm, time_signature).
  2. Build a neutral header prompt for that genre.
  3. Sample one chord continuation from F1 base and one from F1+LoRA, each long
     enough to fill `length_bars × 2` slot chords.
  4. Pack chord names into `bars: list[{chords: [c0, c1]}]` and call the inline
     port of `song_builder.build_*` to render harmony/bass/drum events.
  5. Compute the 7 metrics per (genre, model, prompt) record below.

Output:
  ai/results/r3_arrangement_ablation.csv — one row per (genre, model)
  ai/results/r3_arrangement_ablation.md  — table per metric + aggregate footer

The markdown explicitly notes that these are sensitivity metrics, not quality
metrics — arrangement quality requires the §5 listening study which is
deferred.

Usage (executed after R2):
  cd ai
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/r3_arrangement_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

# decoding_artifacts already lives in ai/training/ and gives us the canonical
# F1 / LoRA loaders + the multinomial sampler.
from decoding_artifacts import (
    _ckpt_dir_for,
    _generate_batch,
    _load_f1,
    _load_lora,
    _selected_ranks,
)
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Mirror of the 11-LoRA list from decoding_artifacts.py / aggregate_multi_seed.py.
GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]

# ---------------------------------------------------------------------------
# Inline port of the bits we need from backend/app/services/{genre_tables,
# song_builder, voice_leading}. The backend versions live behind a FastAPI
# package boundary (`from app.services...` imports + `mido` for MIDI render
# + pydantic schemas in voice_leading.apply), so we re-state the rhythm /
# pattern / chord tables here verbatim. Keep them in sync with the backend
# tables; both sides are deliberately small. Skipped from the port:
#   - mido MIDI render (we only need symbolic event counts)
#   - DRUM_VOICE_TO_GM_NOTE / GM program tables (renderer-side)
#   - voice_leading.apply_voice_leading_bass (operates on Suggestion schema)
# ---------------------------------------------------------------------------

# Per-genre 1-bar harmony comping rhythm — beat positions in [0, beats_per_measure).
GENRE_HARMONY_RHYTHM: dict[str, list[float]] = {
    "pop":        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    "country":    [1, 3],
    "rock":       [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    "blues":      [0, 2/3, 1, 5/3, 2, 8/3, 3, 11/3],
    "bossa":      [0, 1.5, 3],
    "classical":  [0, 1, 2, 3],
    "folk":       [0, 2, 2.5, 3, 3.5],
    "rnb_soul":   [1.5, 2.5, 3.5],
    "hip_hop":    [0, 2.5],
    "electronic": [0.5, 1.5, 2.5, 3.5],
    "funk":       [0, 0.5, 2, 2.5],
    "gospel":     [1, 3],
    "jazz":       [0, 1.5],
}

GENRE_BASS_RHYTHM: dict[str, list[float]] = {
    "pop":        [0, 1, 2, 3],
    "country":    [0, 2],
    "rock":       [0, 1, 2, 3],
    "blues":      [0, 1, 2, 3],
    "bossa":      [0, 1.5],
    "classical":  [0],
    "folk":       [0, 1, 2, 3],
    "rnb_soul":   [0, 1, 2, 3],
    "hip_hop":    [0, 3],
    "electronic": [0, 1, 2, 3],
    "funk":       [0, 0.5, 2, 2.5],
    "gospel":     [0, 2],
    "jazz":       [0, 1, 2, 3],
}

# Per-genre default key/bpm/time-signature (matches genre_tables.GENRE_DEFAULTS).
GENRE_DEFAULTS: dict[str, dict] = {
    "jazz":       {"key": "F major",  "bpm": 110, "time_signature": (4, 4)},
    "pop":        {"key": "C major",  "bpm": 110, "time_signature": (4, 4)},
    "rock":       {"key": "E major",  "bpm": 120, "time_signature": (4, 4)},
    "blues":      {"key": "E major",  "bpm": 90,  "time_signature": (4, 4)},
    "bossa":      {"key": "F major",  "bpm": 110, "time_signature": (4, 4)},
    "classical":  {"key": "C major",  "bpm": 80,  "time_signature": (4, 4)},
    "country":    {"key": "G major",  "bpm": 100, "time_signature": (4, 4)},
    "rnb_soul":   {"key": "Bb major", "bpm": 90,  "time_signature": (4, 4)},
    "hip_hop":    {"key": "A minor",  "bpm": 90,  "time_signature": (4, 4)},
    "electronic": {"key": "A minor",  "bpm": 128, "time_signature": (4, 4)},
    "funk":       {"key": "F major",  "bpm": 110, "time_signature": (4, 4)},
    "folk":       {"key": "G major",  "bpm": 100, "time_signature": (4, 4)},
    "gospel":     {"key": "F major",  "bpm": 90,  "time_signature": (4, 4)},
}

# Drum patterns — only voices/hits matter for event counts. Mirror of
# genre_tables.GENRE_DRUM_PATTERN.
GENRE_DRUM_PATTERN: dict[str, dict] = {
    "pop": {
        "voices": {
            "C": [[0, 0.5]],
            "K": [[0, 1.0], [6, 0.9], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [7, 0.28], [11, 0.28], [14, 0.32]],
            "H": [[0, 0.95], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.95], [10, 0.55], [12, 0.85]],
            "O": [[14, 0.65]],
        },
    },
    "hip_hop": {
        "voices": {
            "K": [[0, 1.0], [11, 0.9]],
            "S": [[8, 1.0], [2, 0.3], [3, 0.22], [6, 0.28], [10, 0.35], [11, 0.22], [13, 0.28], [14, 0.35], [15, 0.28]],
            "H": [[0, 0.9], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.9], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    "electronic": {
        "voices": {
            "C": [[0, 0.6]],
            "K": [[0, 1.0], [4, 1.0], [8, 1.0], [12, 1.0], [3, 0.6]],
            "S": [[4, 1.0], [12, 1.0], [10, 0.45], [14, 0.5]],
            "H": [[0, 0.95], [1, 0.5], [2, 0.7], [3, 0.5], [4, 0.95], [5, 0.5], [6, 0.7], [7, 0.85],
                  [8, 0.95], [9, 0.5], [10, 0.7], [11, 0.85], [12, 0.95], [13, 0.5], [14, 0.7], [15, 0.85]],
            "O": [[2, 0.7], [6, 0.7], [10, 0.7], [14, 0.7]],
            "X": [[7, 0.6], [15, 0.6]],
        },
    },
    "jazz": {
        "voices": {
            "R": [[0, 1.0], [4, 0.9], [6, 0.7], [8, 1.0], [12, 0.9], [14, 0.7]],
            "H": [[4, 0.55], [12, 0.55]],
            "S": [[2, 0.18], [10, 0.22]],
            "K": [[0, 0.55], [8, 0.5]],
        },
    },
    "bossa": {
        "voices": {
            "K": [[0, 1.0], [6, 0.85], [8, 1.0], [14, 0.85]],
            "X": [[3, 0.9], [6, 0.9], [10, 0.9], [12, 0.9]],
            "H": [[0, 0.75], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.75], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    "folk": {
        "voices": {
            "K": [[0, 0.9], [10, 0.7]],
            "X": [[8, 0.85]],
            "H": [[0, 0.65], [4, 0.65], [8, 0.65], [12, 0.65]],
        },
    },
    "funk": {
        "voices": {
            "K": [[0, 1.0], [3, 0.85], [7, 0.8], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [6, 0.22], [11, 0.22], [14, 0.22]],
            "H": [[0, 0.85], [1, 0.4], [2, 0.65], [3, 0.4], [4, 0.85], [5, 0.4], [6, 0.65], [7, 0.4],
                  [8, 0.85], [9, 0.4], [10, 0.65], [11, 0.4], [12, 0.85], [13, 0.4], [14, 0.65], [15, 0.4]],
        },
    },
    "blues": {
        "voices": {
            "K": [[0, 1.0], [8, 1.0], [10, 0.7]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.85], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.85], [10, 0.55], [12, 0.85], [14, 0.55]],
        },
    },
    "rock": {
        "voices": {
            "C": [[0, 1.0]],
            "K": [[0, 1.0], [3, 0.9], [8, 1.0], [11, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [6, 0.25]],
            "H": [[0, 0.95], [2, 0.6], [4, 0.9], [6, 0.6], [8, 0.95], [10, 0.6]],
            "T": [[12, 1.0]],
            "M": [[13, 0.95]],
            "L": [[14, 0.95], [15, 1.0]],
        },
    },
    "country": {
        "voices": {
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "X": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
            "H": [[0, 0.7], [4, 0.7], [8, 0.7], [12, 0.7]],
        },
    },
    "rnb_soul": {
        "voices": {
            "K": [[0, 1.0], [10, 0.85]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [3, 0.18], [6, 0.24], [10, 0.26], [14, 0.26], [15, 0.18]],
            "H": [[0, 0.85], [2, 0.5], [4, 0.75], [6, 0.5], [8, 0.85], [10, 0.5], [12, 0.75], [14, 0.5]],
        },
    },
    "gospel": {
        "voices": {
            "C": [[0, 0.45]],
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.8], [4, 0.8], [8, 0.8], [12, 0.8]],
            "O": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
        },
    },
    "classical": {
        "voices": {
            "L": [[0, 0.6]],
            "C": [[0, 0.3]],
        },
    },
}

# Inline copy of voice_leading._NOTE_PC and _CHORD_TONES so we can parse chord
# symbols + know voicing sizes without importing from the FastAPI backend.
_NOTE_PC: dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1, "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "F": 5,  "F#": 6, "Gb": 6, "G": 7,  "G#": 8,
    "Ab": 8, "A": 9,  "A#": 10, "Bb": 10, "B": 11,
}

# Diatonic pitch-class sets for the major/minor scale starting from each root.
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor


def _parse_chord(sym: str) -> tuple[int, str, int | None] | None:
    """Port of voice_leading._parse — return (root_pc, quality, bass_pc) or None."""
    if not sym:
        return None
    s = sym.strip()
    bass_pc: int | None = None
    if "/" in s:
        s, bass_str = s.split("/", 1)
        bass_pc = _NOTE_PC.get(bass_str.strip())
    if len(s) >= 2 and s[1] in ("#", "b"):
        root, quality = s[:2], s[2:]
    elif s:
        root, quality = s[:1], s[1:]
    else:
        return None
    root_pc = _NOTE_PC.get(root)
    if root_pc is None:
        return None
    return root_pc, (quality or "maj"), bass_pc


def _slot_chord(bar_chords: list[str | None], pos: float, beats_per_slot: float) -> str | None:
    if not bar_chords:
        return None
    slot_idx = 0 if pos < beats_per_slot else 1
    if slot_idx >= len(bar_chords):
        slot_idx = len(bar_chords) - 1
    return bar_chords[slot_idx]


def _key_to_diatonic_pcs(key_str: str) -> set[int]:
    """Convert 'F major' / 'A minor' / 'Bb major' to its diatonic pitch-class set."""
    parts = key_str.strip().split()
    if not parts:
        return set()
    root = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    root_pc = _NOTE_PC.get(root)
    if root_pc is None:
        return set()
    scale = _MINOR_SCALE if mode.startswith("min") else _MAJOR_SCALE
    return {(root_pc + iv) % 12 for iv in scale}


# ---------------------------------------------------------------------------
# Arrangement rendering: chord stream → (harmony, bass, drum) event counts
# ---------------------------------------------------------------------------


def build_harmony_events(bars: list[dict], genre: str, *, beats_per_measure: int) -> list[dict]:
    """Port of song_builder.build_harmony_events but emits a single event per
    pattern-position (we don't need the per-voice expansion to count events)."""
    pattern = GENRE_HARMONY_RHYTHM.get(genre, [0, 2])
    beats_per_slot = beats_per_measure / 2
    events: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords = bar.get("chords") or []
        for pos in pattern:
            if pos < 0 or pos >= beats_per_measure:
                continue
            chord_sym = _slot_chord(chords, pos, beats_per_slot)
            if not chord_sym:
                continue
            if _parse_chord(chord_sym) is None:
                continue
            events.append({"bar": bar_idx, "beat": float(pos)})
    return events


def build_bass_events(bars: list[dict], genre: str, *, beats_per_measure: int) -> list[dict]:
    pattern = GENRE_BASS_RHYTHM.get(genre, [0, 1, 2, 3])
    beats_per_slot = beats_per_measure / 2
    events: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords = bar.get("chords") or []
        for pos in pattern:
            if pos < 0 or pos >= beats_per_measure:
                continue
            chord_sym = _slot_chord(chords, pos, beats_per_slot)
            if not chord_sym:
                continue
            if _parse_chord(chord_sym) is None:
                continue
            events.append({"bar": bar_idx, "beat": float(pos)})
    return events


def build_drum_events(n_bars: int, genre: str, *, beats_per_measure: int) -> list[dict]:
    p = GENRE_DRUM_PATTERN.get(genre)
    if p is None:
        return []
    events: list[dict] = []
    for bar_idx in range(n_bars):
        for _voice, hits in p["voices"].items():
            for hit in hits:
                pos16, _vel = hit
                if pos16 >= 16:
                    continue
                events.append({"bar": bar_idx, "pos16": int(pos16)})
    return events


# ---------------------------------------------------------------------------
# Chord-token extraction + chord-stream → bars packing
# ---------------------------------------------------------------------------


def _is_chord_token(tok: str) -> bool:
    """Same gate r1/r2 use — anything not starting with [ or < is chord-shaped."""
    return bool(tok) and not tok.startswith(("[", "<"))


def _extract_chord_strs(sample_ids: list[int], prompt_len: int, tok: ChordTokenizer) -> list[str]:
    """Drop the header prompt and pull only chord-shaped continuation tokens.
    Stops at [EOS] (decoded as '[EOS]' by the tokenizer)."""
    out: list[str] = []
    for tid in sample_ids[prompt_len:]:
        s = tok.id2token.get(int(tid), "")
        if s == "[EOS]":
            break
        if _is_chord_token(s):
            out.append(s)
    return out


def _pack_bars(chords: list[str], length_bars: int) -> list[dict]:
    """Pack a flat chord list into `length_bars` bars of 2 slots each. Pads with
    the last available chord if generation came up short (defensive — most
    samples produce more than enough chords)."""
    needed = length_bars * 2
    if not chords:
        return []
    if len(chords) < needed:
        chords = chords + [chords[-1]] * (needed - len(chords))
    bars: list[dict] = []
    for b in range(length_bars):
        c0 = chords[2 * b]
        c1 = chords[2 * b + 1]
        bars.append({"chords": [c0, c1]})
    return bars


def _build_prompt(tok: ChordTokenizer, genre: str, key_str: str, time_sig: tuple[int, int]) -> list[int]:
    """Header prompt that matches the genre's default key + time signature so
    the model sees realistic conditioning (instead of always Cmaj 4/4)."""
    # Tokenizer wants `Cmaj` / `Bbmaj` / `Amin` (no space); convert from
    # GENRE_DEFAULTS' human spelling.
    root, mode = key_str.split()
    mode_key = "maj" if mode.lower().startswith("maj") else "min"
    key_token = f"{root}{mode_key}"
    ts_token = f"{time_sig[0]}/{time_sig[1]}"
    out = [tok.bos_id]
    for special in (
        tok.encode_key(key_token),
        tok.encode_time_sig(ts_token),
        tok.encode_genre(genre),
    ):
        if special is not None:
            out.append(special)
    out.append(tok.bar_id)
    return out


# ---------------------------------------------------------------------------
# Metric computation — the 7 arrangement-axis metrics from the spec.
# Metrics 2 (bass) and 3 (drum) are reported for completeness; they would
# only change if the chord stream causes some bars to be all-unparseable.
# ---------------------------------------------------------------------------


def _compute_metrics(
    chord_stream: list[str],
    bars: list[dict],
    genre: str,
    key_str: str,
    beats_per_measure: int,
) -> dict:
    """Compute the 7 arrangement-axis metrics. All counts are per-bar means."""
    n_bars = max(1, len(bars))

    harmony_events = build_harmony_events(bars, genre, beats_per_measure=beats_per_measure)
    bass_events = build_bass_events(bars, genre, beats_per_measure=beats_per_measure)
    drum_events = build_drum_events(n_bars, genre, beats_per_measure=beats_per_measure)

    # 1-3: per-bar event counts.
    harmony_per_bar = len(harmony_events) / n_bars
    bass_per_bar = len(bass_events) / n_bars
    drum_per_bar = len(drum_events) / n_bars

    # 4: chord_change_density — distinct chord symbols per bar (mean).
    per_bar_distinct = []
    for bar in bars:
        chords = [c for c in (bar.get("chords") or []) if c]
        per_bar_distinct.append(len(set(chords)))
    chord_change_density = (
        sum(per_bar_distinct) / max(1, len(per_bar_distinct)) if per_bar_distinct else 0.0
    )

    # 5: unique_chord_qualities across the whole rendered song.
    qualities = set()
    roots_pc: list[int] = []
    for c in chord_stream:
        parsed = _parse_chord(c)
        if parsed is None:
            continue
        root_pc, quality, _ = parsed
        qualities.add(quality)
        roots_pc.append(root_pc)
    unique_chord_qualities = len(qualities)

    # 6: non_diatonic_rate — fraction of chord roots outside the key's diatonic set.
    diatonic_pcs = _key_to_diatonic_pcs(key_str)
    if roots_pc and diatonic_pcs:
        non_diatonic = sum(1 for pc in roots_pc if pc not in diatonic_pcs)
        non_diatonic_rate = non_diatonic / len(roots_pc)
    else:
        non_diatonic_rate = 0.0

    # 7: harmony_silence_ratio — bars where the comping rhythm fires but no
    # event lands (e.g. both slot-chords were unparseable). Counts per-bar
    # zero-event bars / total bars.
    bars_with_zero_harmony = sum(
        1 for b in range(n_bars)
        if not any(e["bar"] == b for e in harmony_events)
    )
    harmony_silence_ratio = bars_with_zero_harmony / n_bars

    return {
        "harmony_event_count": harmony_per_bar,
        "bass_event_count": bass_per_bar,
        "drum_event_count": drum_per_bar,
        "chord_change_density": chord_change_density,
        "unique_chord_qualities": unique_chord_qualities,
        "non_diatonic_rate": non_diatonic_rate,
        "harmony_silence_ratio": harmony_silence_ratio,
    }


METRIC_KEYS = [
    "harmony_event_count",
    "bass_event_count",
    "drum_event_count",
    "chord_change_density",
    "unique_chord_qualities",
    "non_diatonic_rate",
    "harmony_silence_ratio",
]

# Per-metric notes shown in the markdown so the reader knows which metrics
# are actually sensitive to chord-sequence variation.
METRIC_NOTES = {
    "harmony_event_count": "sensitive — depends on slot-chord parseability",
    "bass_event_count": "near-invariant — fixed rhythm × root strikes",
    "drum_event_count": "invariant per genre — fixed drum pattern × n_bars",
    "chord_change_density": "sensitive — distinct chord symbols / bar",
    "unique_chord_qualities": "sensitive — vocabulary breadth",
    "non_diatonic_rate": "sensitive — out-of-key chord rate vs genre default key",
    "harmony_silence_ratio": "sensitive — fraction of bars that comp nothing",
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summarise(metrics_records: list[dict]) -> dict:
    """Mean + stdev across the per-prompt metric records for one (genre, model)."""
    if not metrics_records:
        return {k: 0.0 for k in METRIC_KEYS} | {f"{k}_std": 0.0 for k in METRIC_KEYS}
    out: dict[str, float] = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in metrics_records]
        out[k] = round(statistics.mean(vals), 3)
        out[f"{k}_std"] = round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts-per-genre", type=int, default=10,
                        help="Number of sampled continuations per (genre, model)")
    parser.add_argument("--length-bars", type=int, default=8,
                        help="Bars to render per sample (= 2 chord slots each)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generation budget per sample (must produce ≥ 2×length_bars chord tokens)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(
        f"Config: n_prompts={args.n_prompts_per_genre} length_bars={args.length_bars} "
        f"temp={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens}"
    )

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    selected = _selected_ranks()

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    rows: list[dict] = []

    for i, genre in enumerate(GENRES, start=1):
        rank = selected.get(genre)
        if rank is None:
            print(f"[{i}/{len(GENRES)}] {genre}: SKIP — no selected rank")
            continue
        ckpt_dir = _ckpt_dir_for(genre, rank)
        if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
            print(f"[{i}/{len(GENRES)}] {genre} r={rank}: SKIP — adapter missing at {ckpt_dir}")
            continue

        defaults = GENRE_DEFAULTS.get(genre, GENRE_DEFAULTS["pop"])
        key_str = defaults["key"]
        time_sig = defaults["time_signature"]
        beats_per_measure = int(time_sig[0])
        prompt = _build_prompt(new_tok, genre, key_str, time_sig)
        prompt_len = len(prompt)
        print(
            f"[{i}/{len(GENRES)}] {genre} r={rank} key={key_str} ts={time_sig[0]}/{time_sig[1]} "
            f"prompt_len={prompt_len}"
        )

        # F1 baseline samples
        f1_seqs = _generate_batch(
            f1, prompt, args.n_prompts_per_genre, args.max_new_tokens,
            new_tok.eos_id, args.temperature, args.top_p, device,
        )
        f1_records: list[dict] = []
        for ids in f1_seqs:
            chord_strs = _extract_chord_strs(ids, prompt_len, new_tok)
            bars = _pack_bars(chord_strs, args.length_bars)
            if not bars:
                continue
            f1_records.append(_compute_metrics(
                chord_strs[: args.length_bars * 2], bars, genre, key_str, beats_per_measure,
            ))
        f1_summary = _summarise(f1_records)
        rows.append({
            "genre": genre, "rank": rank, "model": "F1",
            "n_samples": len(f1_records), **f1_summary,
        })

        # LoRA samples
        lora = _load_lora(genre, rank, old_tok, new_tok, device)
        lora_seqs = _generate_batch(
            lora, prompt, args.n_prompts_per_genre, args.max_new_tokens,
            new_tok.eos_id, args.temperature, args.top_p, device,
        )
        lora_records: list[dict] = []
        for ids in lora_seqs:
            chord_strs = _extract_chord_strs(ids, prompt_len, new_tok)
            bars = _pack_bars(chord_strs, args.length_bars)
            if not bars:
                continue
            lora_records.append(_compute_metrics(
                chord_strs[: args.length_bars * 2], bars, genre, key_str, beats_per_measure,
            ))
        lora_summary = _summarise(lora_records)
        rows.append({
            "genre": genre, "rank": rank, "model": "LoRA",
            "n_samples": len(lora_records), **lora_summary,
        })
        del lora
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Print per-genre deltas for quick eyeballing in stdout.
        deltas = " | ".join(
            f"{k}={lora_summary[k] - f1_summary[k]:+.3f}" for k in METRIC_KEYS
        )
        print(f"  → Δ(LoRA−F1) per metric: {deltas}")

    if not rows:
        print("No rows produced — every genre was skipped. Exiting.")
        return

    # ---- CSV ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["genre", "rank", "model", "n_samples"] + METRIC_KEYS + [f"{k}_std" for k in METRIC_KEYS]
    out_csv = RESULTS_DIR / "r3_arrangement_ablation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    # ---- Markdown ----
    by_g: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_g.setdefault(r["genre"], {})[r["model"]] = r

    lines = [
        "# §3.2 R3 rule-based arrangement axis ablation",
        "",
        "**Question:** Does swapping F1 → F1+LoRA at the chord layer produce "
        "measurably different arrangement renders downstream?",
        "",
        "**Setup:** The deployed multi-track pipeline is:",
        "",
        "    chord stream  ──►  song_builder.build_*  ──►  {harmony, bass, drum} events",
        "",
        "Arrangement is purely rule-based and pulled from per-genre tables in "
        "`backend/app/services/genre_tables.py` (`GENRE_HARMONY_RHYTHM` / "
        "`GENRE_BASS_RHYTHM` / `GENRE_DRUM_PATTERN`). Every knob below the chord "
        "stream is a genre invariant — so any difference between F1 and F1+LoRA "
        "renders comes entirely from the chord sequence the model proposes.",
        "",
        f"**Protocol:** for each genre, sample `n={args.n_prompts_per_genre}` "
        f"continuations from F1 base and from F1 + selected-rank LoRA from a header "
        f"prompt matching that genre's default key / time signature "
        f"(`GENRE_DEFAULTS`). Decode the first `length_bars × 2 = "
        f"{args.length_bars * 2}` chord-shaped tokens into 2-slot bars, then run "
        f"the inline port of `song_builder.build_*` to render harmony / bass / "
        f"drum events. Decoding: temperature={args.temperature}, "
        f"top_p={args.top_p}, max_new_tokens={args.max_new_tokens}.",
        "",
        "**Honest framing:** these are **arrangement-side sensitivity metrics**, "
        "not arrangement-quality metrics. They tell us whether LoRA's chord "
        "distribution moves the downstream render — not whether the render sounds "
        "better. Quality requires the §5 listening study, deferred to the next "
        "cycle.",
        "",
        "Metrics:",
        "",
    ]
    for k in METRIC_KEYS:
        lines.append(f"- **{k}** — {METRIC_NOTES[k]}")
    lines.append("")

    # One table per metric showing F1 / LoRA / Δ per genre.
    for k in METRIC_KEYS:
        lines += [
            f"## {k}",
            "",
            f"_{METRIC_NOTES[k]}_",
            "",
            "| Genre | Rank | n | F1 | LoRA | Δ (LoRA − F1) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        deltas: list[float] = []
        for g in GENRES:
            cells = by_g.get(g)
            if not cells:
                continue
            f1c = cells.get("F1")
            lc = cells.get("LoRA")
            if not (f1c and lc):
                continue
            f1_val = f1c[k]
            lora_val = lc[k]
            d = lora_val - f1_val
            deltas.append(d)
            lines.append(
                f"| {g} | {f1c['rank']} | {lc['n_samples']} | "
                f"{f1_val:.3f} | {lora_val:.3f} | {d:+.3f} |"
            )
        if deltas:
            mean_d = statistics.mean(deltas)
            median_d = statistics.median(deltas)
            lines += [
                "",
                f"Across {len(deltas)} genres: mean Δ = {mean_d:+.3f}, "
                f"median Δ = {median_d:+.3f}, "
                f"min Δ = {min(deltas):+.3f}, max Δ = {max(deltas):+.3f}.",
                "",
            ]
        else:
            lines += ["", "_(no usable rows)_", ""]

    # Aggregate footer.
    lines += [
        "## Aggregate (mean Δ across genres, per metric)",
        "",
        "| Metric | Mean Δ (LoRA − F1) | Note |",
        "|---|---:|---|",
    ]
    for k in METRIC_KEYS:
        deltas = []
        for g in GENRES:
            cells = by_g.get(g)
            if not cells:
                continue
            f1c = cells.get("F1")
            lc = cells.get("LoRA")
            if f1c and lc:
                deltas.append(lc[k] - f1c[k])
        if deltas:
            lines.append(
                f"| {k} | {statistics.mean(deltas):+.3f} | {METRIC_NOTES[k]} |"
            )
        else:
            lines.append(f"| {k} | -- | {METRIC_NOTES[k]} |")

    lines += [
        "",
        "## Interpretation guide",
        "",
        "- Non-zero Δ on a **sensitive** metric (chord_change_density / "
        "unique_chord_qualities / non_diatonic_rate / harmony_silence_ratio / "
        "harmony_event_count) ⇒ LoRA's chord distribution is producing a "
        "different downstream arrangement footprint than F1 base.",
        "- Near-zero Δ on **invariant** metrics (bass_event_count / "
        "drum_event_count) is expected — these depend on rule-of-genre rhythm "
        "tables. Non-trivial Δ here would indicate one model is generating more "
        "unparseable chords (silencing bass slots).",
        "- These metrics do NOT score arrangement *quality*; they score "
        "whether the chord layer's identity propagates downstream at all. "
        "For perceptual quality see §5 listening study (deferred).",
    ]

    out_md = RESULTS_DIR / "r3_arrangement_ablation.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    sys.exit(main() or 0)
