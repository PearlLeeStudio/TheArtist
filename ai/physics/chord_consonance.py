"""Chord-level Sethares roughness scoring.

Maps a MIDI chord voicing to a single roughness scalar by summing the
Sethares pair-roughness over all distinct note pairs in the chord, with
each note represented by its harmonic-series partials.

Lower scores = more consonant (closer to a single periodic complex tone).
"""
from __future__ import annotations

import numpy as np

from .dissonance import cross_roughness


def midi_to_freq(midi: float) -> float:
    """Equal-tempered MIDI number → frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def harmonic_partials(
    fundamental: float,
    n_partials: int = 7,
    rolloff: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (frequencies, amplitudes) for a harmonic complex tone.

    Partial k (1-indexed) sits at k * f0 with amplitude 1 / k**rolloff.
    Default rolloff=1.0 gives the canonical 1, 1/2, 1/3, ... harmonic
    spectrum of an idealised periodic source.
    """
    k = np.arange(1, n_partials + 1, dtype=float)
    return fundamental * k, 1.0 / k**rolloff


def chord_roughness(
    midi_notes: list[float],
    n_partials: int = 7,
    rolloff: float = 1.0,
) -> float:
    """Total Sethares roughness for a MIDI chord voicing.

    Sums cross_roughness over every distinct note pair; identical notes
    contribute 0 (their partials coincide).
    """
    if len(midi_notes) < 2:
        return 0.0
    partials = [
        harmonic_partials(midi_to_freq(m), n_partials, rolloff)
        for m in midi_notes
    ]
    total = 0.0
    for i in range(len(partials)):
        fi, ai = partials[i]
        for j in range(i + 1, len(partials)):
            fj, aj = partials[j]
            total += cross_roughness(fi, ai, fj, aj)
    return total


def transition_roughness(
    voicing_a: list[float],
    voicing_b: list[float],
    n_partials: int = 7,
    rolloff: float = 1.0,
) -> float:
    """Mean of the two endpoint chord roughnesses.

    A first-pass scoring rule for chord transitions. More elaborate rules
    (voice-leading distance, common-tone retention) can be layered on top
    in a re-rank function.
    """
    return 0.5 * (
        chord_roughness(voicing_a, n_partials, rolloff)
        + chord_roughness(voicing_b, n_partials, rolloff)
    )
