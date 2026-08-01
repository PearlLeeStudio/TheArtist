"""Stolzenburg (2015) periodicity model for chord harmonicity.

Complements `chord_consonance.chord_roughness` (Sethares sensory axis) with
a periodicity axis — how close a chord's pitch-class set comes to a single
periodic complex tone.

Intuition: if every note in a chord can be expressed as a low-integer-ratio
partial of a common fundamental, the chord is *harmonic* in the strict
physical sense (all upper partials reinforce a single virtual pitch). If
the approximation requires large integer ratios, the chord's partials do
not align with any common fundamental and the chord is *inharmonic*.

Stolzenburg's relative periodicity is computed by:

    1. Reduce notes to pitch classes mod 12 and compute intervals from the
       lowest pitch class.
    2. Look up each interval's just-intonation ratio from the standard
       semitone table (see `_JI_DENOM` below).
    3. Return the LCM of those denominators. The smaller it is, the
       closer the chord is to a single periodic complex.

Lower score = more harmonious. Complements Sethares roughness: pure
Sethares ranks Major 3rd as rougher than Tritone (correct for
psychoacoustic interference, at odds with Western tonal practice);
periodicity ranks Major 3rd (period 4) sharply below Tritone (period 32),
restoring tonal intuition.

The JI ratios used are the standard 5-limit choices from Stolzenburg
(2015) Table 2:

    0 P1 = 1/1    1 m2 = 16/15   2 M2 = 9/8   3 m3 = 6/5   4 M3 = 5/4
    5 P4 = 4/3    6 TT = 45/32   7 P5 = 3/2   8 m6 = 8/5   9 M6 = 5/3
    10 m7 = 16/9  11 M7 = 15/8   12 P8 = 2/1

Reference:
    Stolzenburg, F. (2015). Harmony perception by periodicity detection.
    Journal of Mathematics and Music, 9(3), 215-238.
    https://doi.org/10.1080/17459737.2015.1033024
"""
from __future__ import annotations

import math

# 5-limit just-intonation ratio denominators for each of the 12 semitones
# above the root (Stolzenburg 2015 Table 2).
_JI_DENOM: dict[int, int] = {
    0:  1,   # 1/1      unison
    1:  15,  # 16/15    minor 2nd
    2:  8,   # 9/8      major 2nd
    3:  5,   # 6/5      minor 3rd
    4:  4,   # 5/4      major 3rd
    5:  3,   # 4/3      perfect 4th
    6:  32,  # 45/32    chromatic tritone
    7:  2,   # 3/2      perfect 5th
    8:  5,   # 8/5      minor 6th
    9:  3,   # 5/3      major 6th
    10: 9,   # 16/9     minor 7th (Pythagorean)
    11: 8,   # 15/8     major 7th
}


def relative_periodicity(midi_notes: list[float]) -> int:
    """Return Stolzenburg's relative periodicity for a chord voicing.

    The chord is reduced to pitch classes mod 12. The lowest pitch class
    serves as the root; each other pitch class's interval is looked up in
    `_JI_DENOM`, and the LCM of those denominators is returned.

    Args:
        midi_notes: list of MIDI numbers (need not be unique or sorted).

    Returns:
        Integer period. Lower is more harmonious. Returns 1 for empty or
        single-pitch-class inputs (unisons, octave doublings).
    """
    pcs = sorted({int(m) % 12 for m in midi_notes})
    if len(pcs) <= 1:
        return 1
    root = pcs[0]
    period = 1
    for pc in pcs:
        period = math.lcm(period, _JI_DENOM[(pc - root) % 12])
    return period


def harmonicity(midi_notes: list[float]) -> float:
    """Logarithmic harmonicity score: ``log2(relative_periodicity)``.

    Smoother for cross-chord comparisons and easier to combine with
    continuous scores like Sethares roughness. Lower = more harmonious.
    Zero for unison, octave, and pitch-class duplicates.
    """
    return math.log2(relative_periodicity(midi_notes))
