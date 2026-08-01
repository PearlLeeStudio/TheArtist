"""Sethares (1993) sensory-dissonance model.

Closed-form roughness function for two pure partials. Used as the primitive
for chord-level consonance scoring in `chord_consonance.py`.

Reference:
    Sethares, W. A. (1993). Local consonance and the relationship between
    timbre and scale. Journal of the Acoustical Society of America, 94(3),
    1218-1228. https://doi.org/10.1121/1.408175

This module captures the "interference" / roughness axis only. For a complete
consonance model that combines roughness with harmonicity and cultural
familiarity, see Harrison & Pearce (2020), `incon`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Sethares (1993) Eqs. 4-5 fitted constants
_B1, _B2 = 3.5, 5.75
_S1, _S2 = 0.0207, 18.96
_D_STAR = 0.24


def pair_roughness(
    f1: float,
    f2: float,
    a1: float = 1.0,
    a2: float = 1.0,
) -> float:
    """Sethares roughness contribution of a single partial pair.

    Returns 0 when the partials coincide, peaks roughly one critical band
    apart, and decays exponentially beyond. Symmetric in (f1, f2).

    Args:
        f1, f2: partial frequencies in Hz.
        a1, a2: linear partial amplitudes.

    Returns:
        Roughness in arbitrary Sethares units.
    """
    fmin = min(f1, f2)
    fdiff = abs(f2 - f1)
    s = _D_STAR / (_S1 * fmin + _S2)
    return float(min(a1, a2) * (np.exp(-_B1 * s * fdiff) - np.exp(-_B2 * s * fdiff)))


def cross_roughness(
    freqs1: ArrayLike,
    amps1: ArrayLike,
    freqs2: ArrayLike,
    amps2: ArrayLike,
) -> float:
    """Total roughness across all cross-pairs of partials between two complex tones.

    Within-tone partial pairs are excluded — they encode intrinsic timbre
    roughness, which is constant for a fixed timbre and uninformative when
    comparing chord candidates with the same instrument model.
    """
    f1 = np.asarray(freqs1, dtype=float)[:, None]
    f2 = np.asarray(freqs2, dtype=float)[None, :]
    a1 = np.asarray(amps1, dtype=float)[:, None]
    a2 = np.asarray(amps2, dtype=float)[None, :]
    fmin = np.minimum(f1, f2)
    fdiff = np.abs(f2 - f1)
    s = _D_STAR / (_S1 * fmin + _S2)
    pair = np.minimum(a1, a2) * (
        np.exp(-_B1 * s * fdiff) - np.exp(-_B2 * s * fdiff)
    )
    return float(pair.sum())
