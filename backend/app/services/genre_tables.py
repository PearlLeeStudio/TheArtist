"""Per-genre lookup tables — backend mirror of the frontend tables.

These tables drive `/api/generate/song` so voyager can call with
just `(genre, length_bars)` and get back a full multi-track render. Three
layers (harmony / bass / drum); melody dropped 2026-05-10. A request may
also pick a comping rhythm preset, a drum pattern (or none) and a harmony
instrument — the same choices as the editor's Play panel.

**Source of truth — keep in sync:**
- Rhythm presets, bass rhythm, instruments, genre → drum pattern id:
  `frontend/src/engine/genreTables.ts`
- Drum patterns: `frontend/src/engine/drumPatterns.ts`
- Strike lengths (`harmony_strikes`): `harmonyStrikes` in
  `frontend/src/engine/playSettings.ts`

`tests/test_genre_tables_parity.py` loads those files with node and fails
when the two sides drift; edit the frontend, then mirror the change here.
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


# --- Harmony comping rhythm presets (mirror of genreTables.ts) ---
# Beat positions in a 4/4 bar; positions at or past the bar length are
# dropped (3/4). A chord slot with no position landing in it is not struck.
# `genres` = the genres whose default rhythm the preset is; Block and Half
# are pick-only, and Block also plays for a genre outside the 13.

HARMONY_RHYTHM_PRESETS: list[dict] = [
    {"id": "charleston",        "label": "Charleston",        "pattern": [0, 1.5],                          "genres": ["jazz"]},
    {"id": "eighth_pulse",      "label": "8th pulse",         "pattern": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],  "genres": ["pop", "rock"]},
    {"id": "boom_chick",        "label": "Boom-chick",        "pattern": [1, 3],                            "genres": ["country", "gospel"]},
    {"id": "shuffle",           "label": "Shuffle",           "pattern": [0, 2/3, 1, 5/3, 2, 8/3, 3, 11/3], "genres": ["blues"]},
    {"id": "bossa",             "label": "Bossa",             "pattern": [0, 1.5, 3],                       "genres": ["bossa"]},
    {"id": "chorale",           "label": "Chorale",           "pattern": [0, 1, 2, 3],                      "genres": ["classical"]},
    {"id": "folk_pocket",       "label": "Folk pocket",       "pattern": [0, 2, 2.5, 3, 3.5],               "genres": ["folk"]},
    {"id": "neo_soul_offbeat",  "label": "Neo-soul offbeat",  "pattern": [1.5, 2.5, 3.5],                   "genres": ["rnb_soul"]},
    {"id": "boom_bap",          "label": "Boom-bap",          "pattern": [0, 2.5],                          "genres": ["hip_hop"]},
    {"id": "sidechain_offbeat", "label": "Sidechain offbeat", "pattern": [0.5, 1.5, 2.5, 3.5],              "genres": ["electronic"]},
    {"id": "funky_drummer",     "label": "Funky drummer",     "pattern": [0, 0.5, 2, 2.5],                  "genres": ["funk"]},
    {"id": "block",             "label": "Block",             "pattern": [0, 2],                            "genres": []},
    {"id": "half",              "label": "Half time",         "pattern": [0],                               "genres": []},
]

DEFAULT_HARMONY_RHYTHM_ID = "block"

_RHYTHM_PRESET_BY_ID: dict[str, dict] = {p["id"]: p for p in HARMONY_RHYTHM_PRESETS}

# Genre → default comping rhythm, derived from the presets.
GENRE_HARMONY_RHYTHM: dict[str, list[float]] = {
    genre: preset["pattern"] for preset in HARMONY_RHYTHM_PRESETS for genre in preset["genres"]
}


# --- Per-genre 1-bar bass rhythm (mirror of genreTables.ts) ---

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


# --- Per-genre instrument lookup (mirror of genreTables.ts) ---
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
    "gospel":     "drawbar_organ",
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

# The 31 harmony instruments a request may pick instead of the genre
# default, in frontend order. Every genre default is one of them.
HARMONY_INSTRUMENTS: tuple[str, ...] = (
    # keys / voice
    "acoustic_grand_piano", "electric_piano_1", "drawbar_organ", "clavinet",
    "pad_2_warm", "choir_aahs",
    # guitar
    "acoustic_guitar_nylon", "acoustic_guitar_steel", "electric_guitar_clean",
    "overdriven_guitar", "distortion_guitar",
    # strings
    "violin", "viola", "cello", "string_ensemble_1", "pizzicato_strings",
    "orchestral_harp",
    # winds
    "flute", "clarinet", "oboe", "bassoon",
    # reeds
    "soprano_sax", "alto_sax", "tenor_sax", "baritone_sax",
    # brass
    "trumpet", "trombone", "french_horn", "tuba", "brass_section", "muted_trumpet",
)

# What POST /api/generate/song may pick besides the Play-panel choices: the two
# bass instruments, the model's 24 keys spelled as its key tokens are
# (ai/training/tokenizer.py ROOTS), and the tempo range.
BASS_INSTRUMENTS: tuple[str, ...] = ("contrabass", "electric_bass_finger")
SONG_KEY_ROOTS: tuple[str, ...] = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
SONG_KEYS: tuple[str, ...] = tuple(f"{root} {mode}" for mode in ("major", "minor") for root in SONG_KEY_ROOTS)
SONG_BPM_MIN, SONG_BPM_MAX = 40, 220

# General MIDI program (0-based) for the MIDI render's program_change:
# the 31 harmony instruments, then the two bass instruments.
GM_PROGRAM: dict[str, int] = {
    "acoustic_grand_piano":  0,
    "electric_piano_1":      4,
    "clavinet":              7,
    "drawbar_organ":         16,
    "acoustic_guitar_nylon": 24,
    "acoustic_guitar_steel": 25,
    "electric_guitar_clean": 27,
    "overdriven_guitar":     29,
    "distortion_guitar":     30,
    "violin":                40,
    "viola":                 41,
    "cello":                 42,
    "pizzicato_strings":     45,
    "orchestral_harp":       46,
    "string_ensemble_1":     48,
    "choir_aahs":            52,
    "trumpet":               56,
    "trombone":              57,
    "tuba":                  58,
    "muted_trumpet":         59,
    "french_horn":           60,
    "brass_section":         61,
    "soprano_sax":           64,
    "alto_sax":              65,
    "tenor_sax":             66,
    "baritone_sax":          67,
    "oboe":                  68,
    "bassoon":               70,
    "clarinet":              71,
    "flute":                 73,
    "pad_2_warm":            89,
    "contrabass":            32,   # GM Acoustic Bass (upright); GM 43 Contrabass is the bowed string
    "electric_bass_finger":  33,
}


# --- Drum patterns (mirror of drumPatterns.ts DRUM_PATTERNS) ---
# id → {name, kit, swing, genre, voices}. voices: voice letter → list of
# [pos16, velocity] where pos16 is the 16th-note grid index 0..15 within
# one bar. Voice letters: K=kick, S=snare, H=hi-hat closed, O=hi-hat open,
# X=cross-stick/cowbell, C=crash, R=ride, T=tom hi, M=tom mid, L=tom
# low/floor. Funk Show (10) and Samba (11) have no genre: pick-only.
# swing: beats the off-beat 8ths land late in the editor (1/6 = a triplet
# 8th; 0 = straight). The events and MIDI here stay straight.

DRUM_PATTERNS: dict[int, dict] = {
    1: {
        "name": "Pop",
        "kit": "acoustic",
        "swing": 0,
        "genre": "pop",
        "voices": {
            "C": [[0, 0.5]],
            "K": [[0, 1.0], [6, 0.9], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [7, 0.28], [11, 0.28], [14, 0.32]],
            "H": [[0, 0.95], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.95], [10, 0.55], [12, 0.85]],
            "O": [[14, 0.65]],
        },
    },
    2: {
        "name": "Hip-hop",
        "kit": "acoustic",
        "swing": 0,
        "genre": "hip_hop",
        "voices": {
            "K": [[0, 1.0], [11, 0.9]],
            "S": [[8, 1.0], [2, 0.3], [3, 0.22], [6, 0.28], [10, 0.35], [11, 0.22], [13, 0.28], [14, 0.35], [15, 0.28]],
            "H": [[0, 0.9], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.9], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    3: {
        "name": "Electronic",
        "kit": "acoustic",
        "swing": 0,
        "genre": "electronic",
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
    4: {
        "name": "Jazz Swing",
        "kit": "acoustic",
        "swing": 1 / 6,
        "genre": "jazz",
        "voices": {
            "R": [[0, 1.0], [4, 0.9], [6, 0.7], [8, 1.0], [12, 0.9], [14, 0.7]],
            "H": [[4, 0.55], [12, 0.55]],
            "S": [[2, 0.18], [10, 0.22]],
            "K": [[0, 0.55], [8, 0.5]],
        },
    },
    5: {
        "name": "Bossa",
        "kit": "acoustic",
        "swing": 0,
        "genre": "bossa",
        "voices": {
            "K": [[0, 1.0], [6, 0.85], [8, 1.0], [14, 0.85]],
            "X": [[3, 0.9], [6, 0.9], [10, 0.9], [12, 0.9]],
            "H": [[0, 0.75], [2, 0.5], [4, 0.7], [6, 0.5], [8, 0.75], [10, 0.5], [12, 0.7], [14, 0.5]],
        },
    },
    6: {
        "name": "Folk / Ballad",
        "kit": "acoustic",
        "swing": 0,
        "genre": "folk",
        "voices": {
            "K": [[0, 0.9], [10, 0.7]],
            "X": [[8, 0.85]],
            "H": [[0, 0.65], [4, 0.65], [8, 0.65], [12, 0.65]],
        },
    },
    7: {
        "name": "Funk 16",
        "kit": "acoustic",
        "swing": 0,
        "genre": "funk",
        "voices": {
            "K": [[0, 1.0], [3, 0.85], [7, 0.8], [10, 0.9]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [6, 0.22], [11, 0.22], [14, 0.22]],
            "H": [[0, 0.85], [1, 0.4], [2, 0.65], [3, 0.4], [4, 0.85], [5, 0.4], [6, 0.65], [7, 0.4],
                  [8, 0.85], [9, 0.4], [10, 0.65], [11, 0.4], [12, 0.85], [13, 0.4], [14, 0.65], [15, 0.4]],
        },
    },
    8: {
        "name": "Blues Shuffle",
        "kit": "acoustic",
        "swing": 1 / 6,
        "genre": "blues",
        "voices": {
            "K": [[0, 1.0], [8, 1.0], [10, 0.7]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.85], [2, 0.55], [4, 0.85], [6, 0.55], [8, 0.85], [10, 0.55], [12, 0.85], [14, 0.55]],
        },
    },
    9: {
        "name": "Rock Anthem",
        "kit": "acoustic",
        "swing": 0,
        "genre": "rock",
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
    10: {
        "name": "Funk Show",
        "kit": "acoustic",
        "swing": 0,
        "genre": None,
        "voices": {
            "C": [[0, 0.55]],
            "K": [[0, 1.0], [3, 0.9], [6, 0.75], [7, 0.8], [10, 0.9], [13, 0.75]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.25], [6, 0.22], [9, 0.28], [11, 0.25], [14, 0.28], [15, 0.3]],
            "H": [[0, 0.95], [1, 0.45], [2, 0.7], [3, 0.45], [4, 0.9], [5, 0.45], [6, 0.7], [7, 0.45],
                  [8, 0.95], [9, 0.45], [10, 0.7], [11, 0.45], [12, 0.9], [13, 0.45], [14, 0.7]],
            "O": [[15, 0.7]],
        },
    },
    11: {
        "name": "Samba",
        "kit": "acoustic",
        "swing": 0,
        "genre": None,
        "voices": {
            "C": [[0, 0.65]],
            "K": [[0, 1.0], [3, 0.7], [8, 1.0], [11, 0.7]],
            "L": [[4, 0.9], [12, 0.9]],
            "X": [[3, 0.85], [6, 0.85], [10, 0.85], [12, 0.85], [14, 0.85]],
            "H": [[0, 0.8], [1, 0.5], [2, 0.7], [3, 0.5], [4, 0.8], [5, 0.5], [6, 0.7], [7, 0.5],
                  [8, 0.8], [9, 0.5], [10, 0.7], [11, 0.5], [12, 0.8], [13, 0.5], [14, 0.7], [15, 0.5]],
        },
    },
    12: {
        "name": "Country",
        "kit": "acoustic",
        "swing": 0.1,   # a light swing, not a full shuffle
        "genre": "country",
        "voices": {
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "X": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
            "H": [[0, 0.7], [4, 0.7], [8, 0.7], [12, 0.7]],
        },
    },
    13: {
        "name": "R&B / Soul",
        "kit": "acoustic",
        "swing": 0,
        "genre": "rnb_soul",
        "voices": {
            "K": [[0, 1.0], [10, 0.85]],
            "S": [[4, 1.0], [12, 1.0], [2, 0.22], [3, 0.18], [6, 0.24], [10, 0.26], [14, 0.26], [15, 0.18]],
            "H": [[0, 0.85], [2, 0.5], [4, 0.75], [6, 0.5], [8, 0.85], [10, 0.5], [12, 0.75], [14, 0.5]],
        },
    },
    14: {
        "name": "Gospel",
        "kit": "acoustic",
        "swing": 0,
        "genre": "gospel",
        "voices": {
            "C": [[0, 0.45]],
            "K": [[0, 1.0], [8, 1.0]],
            "S": [[4, 1.0], [12, 1.0]],
            "H": [[0, 0.8], [4, 0.8], [8, 0.8], [12, 0.8]],
            "O": [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
        },
    },
    15: {
        "name": "Classical",
        "kit": "acoustic",
        "swing": 0,
        "genre": "classical",
        "voices": {
            "L": [[0, 0.6]],
            "C": [[0, 0.3]],
        },
    },
}

# Genre → its drum pattern id (mirror of genreTables.ts).
GENRE_TO_DRUM_PATTERN_ID: dict[str, int] = {
    "pop":        1,
    "hip_hop":    2,
    "electronic": 3,
    "jazz":       4,
    "bossa":      5,
    "folk":       6,
    "funk":       7,
    "blues":      8,
    "rock":       9,
    "country":    12,
    "rnb_soul":   13,
    "gospel":     14,
    "classical":  15,
}

# Genre → its drum pattern as {name, kit, swing, voices}, derived from the
# two tables above.
GENRE_DRUM_PATTERN: dict[str, dict] = {
    genre: {field: DRUM_PATTERNS[pattern_id][field] for field in ("name", "kit", "swing", "voices")}
    for genre, pattern_id in GENRE_TO_DRUM_PATTERN_ID.items()
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


def harmony_rhythm_preset(preset_id: str | None) -> dict | None:
    """The HARMONY_RHYTHM_PRESETS entry with this id; None when unknown."""
    return _RHYTHM_PRESET_BY_ID.get(preset_id)


def genre_rhythm_preset(genre: str) -> dict:
    """The genre's comping rhythm preset; Block for a genre outside the 13."""
    g = genre.lower()
    for preset in HARMONY_RHYTHM_PRESETS:
        if g in preset["genres"]:
            return preset
    return _RHYTHM_PRESET_BY_ID[DEFAULT_HARMONY_RHYTHM_ID]


def harmony_strikes(pattern: list[float], beats_per_measure: float) -> list[tuple[float, float]]:
    """(position, length) in beats of each strike of a 1-bar rhythm.

    Same formula as the frontend's harmonyStrikes: a strike lasts 0.85 × the
    pattern's average spacing (sparse patterns sustain, dense ones stab),
    clamped to 0.85 × the gap before the next strike — wrapping into the
    next bar — so a chord never runs into its own re-strike. Positions
    outside the bar are dropped; duplicates count once.
    """
    positions = sorted({p for p in pattern if 0 <= p < beats_per_measure})
    if not positions:
        return []
    average = beats_per_measure / len(positions)
    strikes: list[tuple[float, float]] = []
    for i, pos in enumerate(positions):
        nxt = positions[i + 1] if i + 1 < len(positions) else positions[0] + beats_per_measure
        strikes.append((float(pos), 0.85 * min(average, nxt - pos)))
    return strikes


def harmony_pattern(genre: str) -> list[float]:
    return genre_rhythm_preset(genre)["pattern"]


def bass_pattern(genre: str) -> list[float]:
    return GENRE_BASS_RHYTHM.get(genre.lower(), [0, 1, 2, 3])


def harmony_instrument(genre: str) -> str:
    return GENRE_HARMONY_INSTRUMENT.get(genre.lower(), "acoustic_grand_piano")


def bass_instrument(genre: str) -> str:
    return GENRE_BASS_INSTRUMENT.get(genre.lower(), "electric_bass_finger")


def drum_pattern_by_id(pattern_id: int | None) -> dict | None:
    """The DRUM_PATTERNS entry with this id; None when unknown (or None)."""
    return DRUM_PATTERNS.get(pattern_id)


def genre_drum_pattern_id(genre: str) -> int | None:
    """The genre's drum pattern id; None for a genre outside the 13."""
    return GENRE_TO_DRUM_PATTERN_ID.get(genre.lower())


def drum_pattern(genre: str) -> dict | None:
    return GENRE_DRUM_PATTERN.get(genre.lower())


def genre_defaults(genre: str) -> dict:
    """Return {key, bpm, time_signature} for the genre. Falls back to pop."""
    return GENRE_DEFAULTS.get(genre.lower(), GENRE_DEFAULTS["pop"])
