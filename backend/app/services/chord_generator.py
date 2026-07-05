"""Rule-based chord generator with jazz harmony awareness.

Generates 3 suggestions using different harmonic strategies:
  A) Diatonic / circle-of-fifths motion
  B) Tritone substitution
  C) Modal interchange (borrowing from parallel key)
"""

from __future__ import annotations

import random

from app.models.schemas import ContextMeasure, Suggestion

# ── Note helpers ─────────────────────────────────────────────────────────

NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_SEMITONE: dict[str, int] = {}
for _i, _n in enumerate(NOTES_SHARP):
    NOTE_TO_SEMITONE[_n] = _i
for _i, _n in enumerate(NOTES_FLAT):
    NOTE_TO_SEMITONE[_n] = _i

# Keys where sharp note names are preferred (all others default to flats)
# Jazz convention: flats are standard for chord symbols (Ab not G#, Eb not D#)
_SHARP_KEYS = frozenset({"B", "F#", "C#"})


def _note_name(semitone: int, use_flats: bool = True) -> str:
    s = semitone % 12
    return NOTES_FLAT[s] if use_flats else NOTES_SHARP[s]


def _parse_key(key: str) -> tuple[int, bool]:
    """Return (root semitone, is_minor)."""
    parts = key.strip().split()
    root = NOTE_TO_SEMITONE.get(parts[0])
    if root is None:
        raise ValueError(f"Invalid note: {parts[0] if parts else key!r}")
    is_minor = len(parts) > 1 and parts[1].lower().startswith("min")
    return root, is_minor


def _prefer_flats(key_root: str) -> bool:
    """Jazz convention: prefer flats for chord symbols except in very sharp keys."""
    return key_root not in _SHARP_KEYS


# ── Scale & chord tables ────────────────────────────────────────────────

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]  # natural minor

MAJOR_QUALITIES = ["maj7", "m7", "m7", "maj7", "7", "m7", "m7b5"]
MINOR_QUALITIES = ["m7", "m7b5", "maj7", "m7", "m7", "maj7", "7"]

# Genre-aware quality upgrades: degree -> possible upgraded qualities
GENRE_UPGRADES: dict[str, dict[int, list[str]]] = {
    "jazz": {
        0: ["maj9", "6"],       # I -> Imaj9, I6
        1: ["m9"],              # ii -> iim9
        3: ["maj9", "6"],       # IV -> IVmaj9
        4: ["9", "13"],         # V -> V9, V13
    },
    "blues": {
        0: ["9"],               # I -> I9
        3: ["9"],               # IV -> IV9
        4: ["9", "13", "7#9"],  # V -> V9, V7#9
    },
    "bossa nova": {
        0: ["maj9", "6"],
        1: ["m9"],
        4: ["9"],
    },
}


def _diatonic_chord(root: int, degree: int, is_minor: bool, flats: bool,
                    genre: str | None = None) -> str:
    """Build a diatonic chord name, optionally upgraded for genre color."""
    intervals = MINOR_INTERVALS if is_minor else MAJOR_INTERVALS
    qualities = MINOR_QUALITIES if is_minor else MAJOR_QUALITIES
    note = _note_name((root + intervals[degree]) % 12, flats)
    quality = qualities[degree]

    # Genre-specific quality upgrade (~30% chance)
    if genre and genre in GENRE_UPGRADES:
        upgrades = GENRE_UPGRADES[genre].get(degree)
        if upgrades and random.random() < 0.3:
            quality = random.choice(upgrades)

    return f"{note}{quality}" if quality != "maj" else note


# ── Strategy helpers ────────────────────────────────────────────────────

def _pick_weighted(options: list[int], weights: list[float]) -> int:
    return random.choices(options, weights=weights, k=1)[0]


def _collect_preceding_chords(
    context: list[ContextMeasure],
    selected: list[int],
) -> list[str]:
    """Return existing chord names from non-selected measures."""
    return [
        c
        for m in context if m.measure not in selected
        for c in m.chords if c is not None
    ]


def _dominant_of(target_semitone: int, flats: bool) -> str:
    """V7 of target note."""
    return f"{_note_name((target_semitone + 7) % 12, flats)}7"


def _tritone_sub_of(dominant: str, flats: bool) -> str:
    """Tritone substitution: bII7 replaces V7."""
    if not dominant:
        raise ValueError("dominant chord cannot be empty")
    root_name = dominant[0]
    if len(dominant) > 1 and dominant[1] in ("#", "b"):
        root_name = dominant[:2]
    root_semi = NOTE_TO_SEMITONE.get(root_name)
    if root_semi is None:
        raise ValueError(f"Invalid note in dominant: {root_name}")
    return f"{_note_name((root_semi + 6) % 12, flats)}7"


# ── Generation strategies ───────────────────────────────────────────────

StrategyResult = tuple[str, list[str]]


def _tonic_chord(root: int, is_minor: bool, flats: bool) -> str:
    """Return the tonic chord (I or i) with appropriate quality."""
    note = _note_name(root, flats)
    return f"{note}m7" if is_minor else f"{note}maj7"


def _cadence_ending(root: int, is_minor: bool, flats: bool, chords: list[str],
                    position: float) -> list[str]:
    """Apply cadence pattern to the end of chords based on position in song."""
    if len(chords) < 2:
        return chords

    ii_semi = (root + 2) % 12
    v_semi = (root + 7) % 12

    if position >= 0.85:
        # Near end of song: resolve to tonic (ii-V-I)
        if len(chords) >= 3:
            chords[-3] = f"{_note_name(ii_semi, flats)}m7"
            chords[-2] = f"{_note_name(v_semi, flats)}7"
            chords[-1] = _tonic_chord(root, is_minor, flats)
        else:
            chords[-2] = f"{_note_name(v_semi, flats)}7"
            chords[-1] = _tonic_chord(root, is_minor, flats)
    elif position >= 0.6:
        # Later in song: ii-V (creates tension, expects resolution)
        chords[-2] = f"{_note_name(ii_semi, flats)}m7"
        chords[-1] = f"{_note_name(v_semi, flats)}7"
    # Early in song (< 0.6): no forced cadence, keep variety

    return chords


def _strategy_diatonic(
    root: int, is_minor: bool, flats: bool, n_slots: int,
    genre: str | None, _preceding: list[str], position: float,
) -> StrategyResult:
    """Circle-of-fifths / diatonic voice leading."""
    degrees = list(range(7))
    if not is_minor:
        weights = [3.0, 2.0, 1.0, 3.0, 4.0, 1.5, 0.5]
    else:
        weights = [3.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0]

    chords = [
        _diatonic_chord(root, _pick_weighted(degrees, weights), is_minor, flats, genre)
        for _ in range(n_slots)
    ]

    chords = _cadence_ending(root, is_minor, flats, chords, position)
    return "Diatonic — ii-V voice leading", chords


def _strategy_tritone_sub(
    root: int, is_minor: bool, flats: bool, n_slots: int,
    genre: str | None, _preceding: list[str], position: float,
) -> StrategyResult:
    """Tritone substitution: ii — bII7 instead of ii — V7."""
    chords: list[str] = []

    for i in range(n_slots):
        if i == n_slots - 1:
            v7 = _dominant_of(root, flats)
            chords.append(_tritone_sub_of(v7, flats))
        elif i == n_slots - 2 and n_slots >= 2:
            ii_semi = (root + 2) % 12
            chords.append(f"{_note_name(ii_semi, flats)}m7")
        else:
            fill_degrees = [0, 3, 5] if not is_minor else [0, 2, 5]
            chords.append(_diatonic_chord(root, random.choice(fill_degrees), is_minor, flats, genre))

    # At song end, resolve tritone sub to tonic
    if position >= 0.85 and len(chords) >= 1:
        chords.append(_tonic_chord(root, is_minor, flats))
        chords = chords[-n_slots:]  # trim to fit

    return "Tritone substitution", chords


def _strategy_modal_interchange(
    root: int, is_minor: bool, flats: bool, n_slots: int,
    genre: str | None, _preceding: list[str], position: float,
) -> StrategyResult:
    """Borrow chords from parallel major/minor."""
    parallel = not is_minor
    borrow_degrees = [0, 2, 3, 5, 6]
    home_degrees = [0, 1, 3, 4]

    chords = []
    for _ in range(n_slots):
        if random.random() < 0.5:
            deg = random.choice(borrow_degrees)
            chords.append(_diatonic_chord(root, deg, parallel, flats, genre))
        else:
            deg = random.choice(home_degrees)
            chords.append(_diatonic_chord(root, deg, is_minor, flats, genre))

    chords = _cadence_ending(root, is_minor, flats, chords, position)
    return "Modal interchange", chords


# ── Public API ──────────────────────────────────────────────────────────

_STRATEGIES = [_strategy_diatonic, _strategy_tritone_sub, _strategy_modal_interchange]
_SLOTS_PER_MEASURE = 2


def generate_suggestions(
    key: str,
    genre: str | None,
    context: list[ContextMeasure],
    selected_measures: list[int],
    time_signature: tuple[int, int] = (4, 4),
    position_ratio: float = 0.5,
) -> list[Suggestion]:
    """Generate 3 chord suggestions for selected measures.

    Args:
        position_ratio: Where the selection sits in the song (0.0=start, 1.0=end).
            Affects cadence decisions — near the end, strategies resolve to tonic.
    """
    root, is_minor = _parse_key(key)
    key_root = key.strip().split()[0]
    flats = _prefer_flats(key_root)
    preceding = _collect_preceding_chords(context, selected_measures)
    n_slots = len(selected_measures) * _SLOTS_PER_MEASURE

    suggestions: list[Suggestion] = []
    for idx, strategy in enumerate(_STRATEGIES):
        label, flat_chords = strategy(root, is_minor, flats, n_slots, genre, preceding, position_ratio)

        chord_map: dict[str, list[str]] = {}
        for i, m_idx in enumerate(selected_measures):
            start = i * _SLOTS_PER_MEASURE
            measure_chords = flat_chords[start : start + _SLOTS_PER_MEASURE]
            fallback = flat_chords[-1] if flat_chords else "Cmaj7"
            while len(measure_chords) < _SLOTS_PER_MEASURE:
                measure_chords.append(fallback)
            chord_map[str(m_idx)] = measure_chords

        option_letter = chr(ord("A") + idx)
        suggestions.append(Suggestion(
            label=f"Option {option_letter} — {label}",
            chords=chord_map,
        ))

    return suggestions
