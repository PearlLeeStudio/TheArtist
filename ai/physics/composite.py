"""Composite dissonance — Sethares roughness + Stolzenburg harmonicity.

A simple two-axis blend of the interference and harmonicity terms from
Harrison & Pearce (2020)'s three-axis model. (The third axis,
cultural familiarity, is a corpus statistic we do not compute here.)

The two axes are complementary: Sethares captures short-range
psychoacoustic interference between partials; Stolzenburg captures
whether the chord's pitch-class set can be approximated by a single
periodic complex. Their disagreement is load-bearing — dyads like M3 vs
Tritone flip order between the two measures, and chords like Cdim7
score moderate on Sethares but extreme on Stolzenburg because dim chords
have no common fundamental.

Reference:
    Harrison, P. M. C., & Pearce, M. T. (2020). Simultaneous consonance
    in music perception and composition. Psychological Review, 127(2),
    216-244. https://doi.org/10.1037/rev0000169
    (`harrison2020simultaneous` in the bibliography.)
"""
from __future__ import annotations

from typing import NamedTuple

from .chord_consonance import chord_roughness
from .periodicity import harmonicity


class CompositeScore(NamedTuple):
    roughness: float       # Sethares mean pair roughness
    harmonicity: float     # log2(Stolzenburg periodicity)
    composite: float       # weighted sum (higher = more dissonant)


def composite_dissonance(
    midi_notes: list[float],
    roughness_weight: float = 1.0,
    harmonicity_weight: float = 0.1,
    n_partials: int = 7,
    rolloff: float = 1.0,
) -> CompositeScore:
    """Two-axis composite dissonance.

    Args:
        midi_notes: chord voicing in MIDI numbers.
        roughness_weight, harmonicity_weight: relative contribution of
            each axis to the composite sum. Defaults reflect the rough
            magnitude difference between Sethares roughness (0–2 typical)
            and log-periodicity (0–10 typical); with defaults, a chord
            that doubles its periodicity contributes the same to the
            composite as one whose roughness rises by 0.1.
        n_partials, rolloff: passed to `chord_roughness`.

    Returns:
        CompositeScore tuple. Higher composite = more dissonant overall.
    """
    r = chord_roughness(midi_notes, n_partials=n_partials, rolloff=rolloff)
    h = harmonicity(midi_notes)
    return CompositeScore(
        roughness=r,
        harmonicity=h,
        composite=roughness_weight * r + harmonicity_weight * h,
    )
