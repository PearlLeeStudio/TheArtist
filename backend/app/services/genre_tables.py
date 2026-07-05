"""Per-genre lookup tables — backend mirror of the frontend tables.

These tables drive `/api/generate/song` so voyager can call with
just `(genre, length_bars)` and get back a full multi-track render. Three
layers (harmony / bass / drum); melody dropped 2026-05-10.

**Source of truth — keep in sync:**
- Harmony / bass rhythm + instrument tables: `frontend/src/engine/arrangement.ts`
  + `frontend/src/engine/instruments.ts`
- Drum patterns: `frontend/src/engine/drums.ts`

Drift between the two sides is a real risk; if you edit either side, mirror
the change here. A future refactor could move both to a shared JSON, but
for now the duplication is small and deliberate.
"""

from __future__ import annotations


# --- Genre default key/bpm/time-signature ---
# Voyager sends only `genre` + `length_bars`; artist picks these. Each is a
# musically conventional default for the genre. Pearl 2026-05-10 1차안.

GENRE_DEFAULTS: dict[str, dict] = {
    "jazz":       {"key": "F major", "bpm": 110, "time_signature": (4, 4)},
    "pop":        {"key": "C major", "bpm": 110, "time_signature": (4, 4)},
    "rock":       {"key": "E major", "bpm": 120, "time_signature": (4, 4)},
    "blues":      {"key": "E major", "bpm": 90,  "time_signature": (4, 4)},
    "bossa":      {"key": "F major", "bpm": 110, "time_signature": (4, 4)},
    "classical":  {"key": "C major", "bpm": 80,  "time_signature": (4, 4)},
    "country":    {"key": "G major", "bpm": 100, "time_signature": (4, 4)},
    "rnb_soul":   {"key": "Bb major", "bpm": 90, "time_signature": (4, 4)},
    "hip_hop":    {"key": "A minor", "bpm": 90,  "time_signature": (4, 4)},
    "electronic": {"key": "A minor", "bpm": 128, "time_signature": (4, 4)},
    "funk":       {"key": "F major", "bpm": 110, "time_signature": (4, 4)},
    "folk":       {"key": "G major", "bpm": 100, "time_signature": (4, 4)},
    "gospel":     {"key": "F major", "bpm": 90,  "time_signature": (4, 4)},
}


# --- Per-genre 1-bar harmony comping rhythm (mirror of arrangement.ts) ---
# Beat positions in [0, beatsPerMeasure). Patterns are 1-bar literal; if a
# slot has no position landing in it, that slot's chord simply isn't struck.

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


# --- Per-genre instrument lookup (mirror of instruments.ts) ---
# Voyager renders each track on the named instrument (typical GM Soundfont
# convention: snake_case keys).

GENRE_HARMONY_INSTRUMENT: dict[str, str] = {
    "jazz":       "electric_piano_1",
    "pop":        "acoustic_grand_piano",
    "rock":       "overdriven_guitar",
    "blues":      "electric_guitar_clean",
    "bossa":      "acoustic_guitar_nylon",
    "classical":  "acoustic_grand_piano",
    "country":    "acoustic_guitar_steel",
    "rnb_soul":   "electric_piano_1",
    "hip_hop":    "electric_piano_1",
    "electronic": "pad_2_warm",
    "funk":       "electric_guitar_clean",
    "folk":       "acoustic_guitar_steel",
    "gospel":     "hammond_organ",
}

GENRE_BASS_INSTRUMENT: dict[str, str] = {
    "jazz":       "contrabass",
    "pop":        "electric_bass_finger",
    "rock":       "electric_bass_finger",
    "blues":      "electric_bass_finger",
    "bossa":      "contrabass",
    "classical":  "contrabass",
    "country":    "electric_bass_finger",
    "rnb_soul":   "electric_bass_finger",
    "hip_hop":    "electric_bass_finger",
    "electronic": "electric_bass_finger",
    "funk":       "electric_bass_finger",
    "folk":       "contrabass",
    "gospel":     "electric_bass_finger",
}


# --- Drum patterns (mirror of drums.ts DRUM_PATTERNS) ---
# voices: dict of voice_letter → list of [pos16, velocity] where pos16 is
# the 16th-note grid index 0..15 within one bar. Voice letters: K=kick,
# S=snare, H=hi-hat closed, O=hi-hat open, X=cross-stick/cowbell, C=crash,
# R=ride, T=tom hi, M=tom mid, L=tom low/floor.

GENRE_DRUM_PATTERN: dict[str, dict] = {
    "pop": {
        "name": "Pop",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "C": [[0, 0.5]],
            "K": [[0, 1.0], [6, 0.9], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [7, 0.28], [11, 0.28], [14, 0.32]],
            "H": [[0, 0.95], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.95], [10, 0.55], [12, 0.85]],
            "O": [[14, 0.65]],
        },
    },
    "hip_hop": {
        "name": "Hip-hop",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "K": [[0, 1.0], [11, 0.9]],
            "S": [[8, 1.0], [2, 0.3], [3, 0.22], [6, 0.28], [10, 0.35], [11, 0.22], [13, 0.28], [14, 0.35], [15, 0.28]],
            "H": [[0, 0.9], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.9], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    "electronic": {
        "name": "Electronic",
        "kit": "acoustic",
        "swing": False,
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
        "name": "Jazz Swing",
        "kit": "acoustic",
        "swing": True,
        "voices": {
            "R": [[0, 1.0], [4, 0.9], [6, 0.7], [8, 1.0], [12, 0.9], [14, 0.7]],
            "H": [[4, 0.55], [12, 0.55]],
            "S": [[2, 0.18], [10, 0.22]],
            "K": [[0, 0.55], [8, 0.5]],
        },
    },
    "bossa": {
        "name": "Bossa",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "K": [[0, 1.0], [6, 0.85], [8, 1.0], [14, 0.85]],
            "X": [[3, 0.9], [6, 0.9], [10, 0.9], [12, 0.9]],
            "H": [[0, 0.75], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.75], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    "folk": {
        "name": "Folk / Ballad",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "K": [[0, 0.9], [10, 0.7]],
            "X": [[8, 0.85]],
            "H": [[0, 0.65], [4, 0.65], [8, 0.65], [12, 0.65]],
        },
    },
    "funk": {
        "name": "Funk 16",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "K": [[0, 1.0], [3, 0.85], [7, 0.8], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [6, 0.22], [11, 0.22], [14, 0.22]],
            "H": [[0, 0.85], [1, 0.4], [2, 0.65], [3, 0.4], [4, 0.85], [5, 0.4], [6, 0.65], [7, 0.4],
                  [8, 0.85], [9, 0.4], [10, 0.65], [11, 0.4], [12, 0.85], [13, 0.4], [14, 0.65], [15, 0.4]],
        },
    },
    "blues": {
        "name": "Blues Shuffle",
        "kit": "acoustic",
        "swing": True,
        "voices": {
            "K": [[0, 1.0], [8, 1.0], [10, 0.7]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.85], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.85], [10, 0.55], [12, 0.85], [14, 0.55]],
        },
    },
    "rock": {
        "name": "Rock Anthem",
        "kit": "acoustic",
        "swing": False,
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
        "name": "Country",
        "kit": "acoustic",
        "swing": True,
        "voices": {
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "X": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
            "H": [[0, 0.7], [4, 0.7], [8, 0.7], [12, 0.7]],
        },
    },
    "rnb_soul": {
        "name": "R&B / Soul",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "K": [[0, 1.0], [10, 0.85]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [3, 0.18], [6, 0.24], [10, 0.26], [14, 0.26], [15, 0.18]],
            "H": [[0, 0.85], [2, 0.5], [4, 0.75], [6, 0.5], [8, 0.85], [10, 0.5], [12, 0.75], [14, 0.5]],
        },
    },
    "gospel": {
        "name": "Gospel",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "C": [[0, 0.45]],
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.8], [4, 0.8], [8, 0.8], [12, 0.8]],
            "O": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
        },
    },
    "classical": {
        "name": "Classical",
        "kit": "acoustic",
        "swing": False,
        "voices": {
            "L": [[0, 0.6]],
            "C": [[0, 0.3]],
        },
    },
}


# GM drum-map (channel 10) for MIDI export. Voyager renderers can use these
# OR remap to their own kit. Standard GM percussion programs.
DRUM_VOICE_TO_GM_NOTE: dict[str, int] = {
    "K": 36,   # acoustic bass drum
    "S": 38,   # acoustic snare
    "H": 42,   # closed hi-hat
    "O": 46,   # open hi-hat
    "X": 37,   # side stick (cross-stick)
    "C": 49,   # crash cymbal 1
    "R": 51,   # ride cymbal 1
    "T": 50,   # high tom
    "M": 47,   # low-mid tom
    "L": 41,   # low floor tom
}


def harmony_pattern(genre: str) -> list[float]:
    return GENRE_HARMONY_RHYTHM.get(genre.lower(), [0, 2])


def bass_pattern(genre: str) -> list[float]:
    return GENRE_BASS_RHYTHM.get(genre.lower(), [0, 1, 2, 3])


def harmony_instrument(genre: str) -> str:
    return GENRE_HARMONY_INSTRUMENT.get(genre.lower(), "acoustic_grand_piano")


def bass_instrument(genre: str) -> str:
    return GENRE_BASS_INSTRUMENT.get(genre.lower(), "electric_bass_finger")


def drum_pattern(genre: str) -> dict | None:
    return GENRE_DRUM_PATTERN.get(genre.lower())


def genre_defaults(genre: str) -> dict:
    """Return {key, bpm, time_signature} for the genre. Falls back to pop."""
    return GENRE_DEFAULTS.get(genre.lower(), GENRE_DEFAULTS["pop"])
