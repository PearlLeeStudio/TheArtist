"""Chord-symbol → canonical MIDI voicing.

Bridge between the model's string chord tokens (e.g., `Cmaj7`, `F#m7b5`,
`Am`) and the integer MIDI lists consumed by `chord_consonance`. Uses
music21's `harmony.ChordSymbol` parser; transposes by whole octaves so
the chord root sits in the configured octave (default = octave 4).

Returning `None` for unparsable tokens lets callers skip them rather
than crash on a malformed prediction.
"""
from __future__ import annotations

from music21 import harmony


def voice_chord(chord_token: str, root_octave: int = 4) -> list[int] | None:
    """Parse a chord symbol and return MIDI numbers with the root in `root_octave`.

    Args:
        chord_token: chord symbol such as `Cmaj7`, `Am`, `F#m7b5`.
        root_octave: target octave for the chord root (4 = C4 = MIDI 60).

    Returns:
        List of MIDI numbers in pitch order, or None if music21 cannot parse.
    """
    try:
        cs = harmony.ChordSymbol(chord_token)
        if not cs.pitches or cs.root() is None:
            return None
        target_root_midi = (root_octave + 1) * 12 + (cs.root().midi % 12)
        shift = target_root_midi - cs.root().midi
        return [int(p.midi + shift) for p in cs.pitches]
    except Exception:
        return None


def voice_sequence(
    chord_tokens: list[str], root_octave: int = 4
) -> list[list[int] | None]:
    """Voice each token in a sequence; preserves length with None for failures."""
    return [voice_chord(t, root_octave) for t in chord_tokens]
