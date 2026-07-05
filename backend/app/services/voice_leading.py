"""Voice-leading-aware bass selection for AI-generated chord progressions.

The trained Music Transformer was tokenised with slash-chord bass stripped
(see ``ai/training/tokenizer.py`` "Strip slash-chord bass"), so the model
only ever emits root-position chords. This module re-introduces
inversions as a deterministic post-processing pass: for each
``prev → curr`` transition we enumerate ``curr``'s chord tones as
candidate bass notes, score each by pitch-class distance to the prev
chord's effective bass, and pick the one that minimises bass-line
motion — but only when the saving is musically meaningful.

This is the back-end side of "voice leading" the user described: bass-line
continuity (small steps over leaps), with the 3rd / 5th / 7th of a 7-chord
acting as candidate inversion points.
"""
from __future__ import annotations

from app.models.schemas import Suggestion

# Pitch class for each note name (sharp / flat aware).
_NOTE_PC: dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1, "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "F": 5,  "F#": 6, "Gb": 6, "G": 7,  "G#": 8,
    "Ab": 8, "A": 9,  "A#": 10, "Bb": 10, "B": 11,
}
_PC_FLAT_NAME = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Chord-quality → intervals (semitones from root) of the chord tones that
# can sit in the bass. Tensions (9, 11, 13) are deliberately omitted —
# they're acceptable as voicing colour but rarely as bass notes, even in
# inversions.
_CHORD_TONES: dict[str, tuple[int, ...]] = {
    "maj":    (0, 4, 7),
    "min":    (0, 3, 7),
    "7":      (0, 4, 7, 10),
    "maj7":   (0, 4, 7, 11),
    "m7":     (0, 3, 7, 10),
    "dim":    (0, 3, 6),
    "dim7":   (0, 3, 6, 9),
    "m7b5":   (0, 3, 6, 10),
    "aug":    (0, 4, 8),
    "sus4":   (0, 5, 7),
    "sus2":   (0, 2, 7),
    "6":      (0, 4, 7, 9),
    "m6":     (0, 3, 7, 9),
    "9":      (0, 4, 7, 10),
    "m9":     (0, 3, 7, 10),
    "maj9":   (0, 4, 7, 11),
    "add9":   (0, 4, 7),
    "mMaj7":  (0, 3, 7, 11),
    "11":     (0, 4, 7, 10),
    "m11":    (0, 3, 7, 10),
    "13":     (0, 4, 7, 10),
    "m13":    (0, 3, 7, 10),
    "7b9":    (0, 4, 7, 10),
    "7#9":    (0, 4, 7, 10),
    "7#11":   (0, 4, 7, 10),
    "7b13":   (0, 4, 7, 10),
}


def _parse(sym: str) -> tuple[int, str, int | None] | None:
    """Parse ``Cmaj7`` / ``Bb7`` / ``D/F#`` → ``(root_pc, quality, bass_pc | None)``."""
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


def _effective_bass_pc(sym: str) -> int | None:
    """Resolve a chord symbol to its actual sounding bass pitch class."""
    parsed = _parse(sym)
    if parsed is None:
        return None
    root_pc, _q, bass_pc = parsed
    return bass_pc if bass_pc is not None else root_pc


def _pc_distance(a: int, b: int) -> int:
    """Smallest interval between two pitch classes (semitones, 0..6)."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


def pick_bass_for_smooth_voice_leading(
    prev_sym: str,
    curr_sym: str,
    *,
    min_savings: int = 3,
) -> str | None:
    """Return the bass-note name to use for ``curr_sym`` given ``prev_sym``,
    or ``None`` if root-position is already the best choice.

    Algorithm: enumerate ``curr``'s chord tones as candidate bass notes,
    score each by pitch-class distance from the previous chord's effective
    bass, and pick the closest. Only override the root-position default
    when the saving is at least ``min_savings`` semitones — small wins
    aren't worth the notational complexity of an inversion (musicians
    expect root-position by default).
    """
    prev_bass = _effective_bass_pc(prev_sym)
    parsed = _parse(curr_sym)
    if prev_bass is None or parsed is None:
        return None
    curr_root, curr_quality, _existing_bass = parsed
    intervals = _CHORD_TONES.get(curr_quality)
    if not intervals:
        return None

    # Score every candidate bass = (root + interval) mod 12.
    candidates = [
        (_pc_distance(prev_bass, (curr_root + iv) % 12), (curr_root + iv) % 12)
        for iv in intervals
    ]
    candidates.sort()
    best_d, best_pc = candidates[0]
    root_d = _pc_distance(prev_bass, curr_root)

    if best_pc == curr_root or (root_d - best_d) < min_savings:
        return None
    return _PC_FLAT_NAME[best_pc]


def apply_voice_leading_bass(suggestions: list[Suggestion]) -> list[Suggestion]:
    """Inject inversions across each suggestion's transitions in place.

    Walks every suggestion in measure order; for each transition
    ``prev → curr`` (starting from the second chord), runs
    ``pick_bass_for_smooth_voice_leading`` and rewrites ``curr``'s
    chord symbol to ``Root{quality}/{bass}`` when a smoother bass
    is found. The first chord of each suggestion stays in root
    position (no in-suggestion prev to compare against).
    """
    for sug in suggestions:
        if not sug.chords:
            continue
        sorted_keys = sorted(sug.chords.keys(), key=int)

        # Flat list of (measure_key, slot_idx) so we can mutate by location
        # while walking transitions in temporal order.
        positions: list[tuple[str, int]] = []
        for k in sorted_keys:
            for idx in range(len(sug.chords[k])):
                positions.append((k, idx))

        for i in range(1, len(positions)):
            prev_k, prev_idx = positions[i - 1]
            curr_k, curr_idx = positions[i]
            prev_sym = sug.chords[prev_k][prev_idx]
            curr_sym = sug.chords[curr_k][curr_idx]

            new_bass = pick_bass_for_smooth_voice_leading(prev_sym, curr_sym)
            if new_bass is None:
                continue
            base = curr_sym.split("/", 1)[0]
            sug.chords[curr_k][curr_idx] = f"{base}/{new_bass}"
    return suggestions
