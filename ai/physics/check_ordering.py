"""Sanity check that Sethares chord-roughness ordering matches musical intuition.

Run from repo root::

    cd ai && uv run python -m physics.check_ordering

Expected ordering (lower = more consonant):
    Octave < P5 < M3 < m3 < P4 < M6 < m6 < M2 < m7 < M7 < tritone < m2
    C major < C7 < Cmaj7 < Cm7 < Cm7b5 < Cdim7 < cluster
"""
from __future__ import annotations

import sys

from .chord_consonance import chord_roughness
from .composite import composite_dissonance
from .periodicity import harmonicity, relative_periodicity


def _print_ordered(label: str, items: dict[str, list[int]]) -> None:
    rows = []
    for name, midis in items.items():
        r = chord_roughness(midis)
        p = relative_periodicity(midis)
        h = harmonicity(midis)
        c = composite_dissonance(midis).composite
        rows.append((name, r, p, h, c))
    rows.sort(key=lambda x: x[4])  # sort by composite
    print(f"\n=== {label} (ascending composite) ===")
    print(f"{'name':<32} {'rough':>7} {'period':>7} {'harm':>6} {'comp':>7}")
    print("-" * 64)
    for name, r, p, h, c in rows:
        print(f"{name:<32} {r:>7.3f} {p:>7d} {h:>6.2f} {c:>7.3f}")


def _expect_lt(label: str, a: float, b: float, fails: list[str]) -> None:
    ok = a < b
    print(f"{'OK ' if ok else 'FAIL'}  {label}: {a:.4f} {'<' if ok else '>='} {b:.4f}")
    if not ok:
        fails.append(label)


INTERVALS = {
    "Unison":      [60, 60],
    "Minor 2nd":   [60, 61],
    "Major 2nd":   [60, 62],
    "Minor 3rd":   [60, 63],
    "Major 3rd":   [60, 64],
    "Perfect 4th": [60, 65],
    "Tritone":     [60, 66],
    "Perfect 5th": [60, 67],
    "Minor 6th":   [60, 68],
    "Major 6th":   [60, 69],
    "Minor 7th":   [60, 70],
    "Major 7th":   [60, 71],
    "Octave":      [60, 72],
}

CHORDS = {
    "C major":            [60, 64, 67],
    "C minor":            [60, 63, 67],
    "Cmaj7":              [60, 64, 67, 71],
    "Cm7":                [60, 63, 67, 70],
    "C7":                 [60, 64, 67, 70],
    "Cdim7":              [60, 63, 66, 69],
    "Caug":               [60, 64, 68],
    "Csus4":              [60, 65, 67],
    "Cm7b5":              [60, 63, 66, 70],
    "C-Db-D cluster":     [60, 61, 62],
    "C-D-E whole-tone":   [60, 62, 64],
}


def main() -> int:
    _print_ordered("Dyads from C4", INTERVALS)
    _print_ordered("Common chords", CHORDS)

    print("\n=== Sethares-only ordering assertions ===")
    fails: list[str] = []
    _expect_lt("P5 < tritone",      chord_roughness([60, 67]), chord_roughness([60, 66]), fails)
    _expect_lt("P5 < m2",           chord_roughness([60, 67]), chord_roughness([60, 61]), fails)
    _expect_lt("Octave < tritone",  chord_roughness([60, 72]), chord_roughness([60, 66]), fails)
    _expect_lt("M3 < M2",           chord_roughness([60, 64]), chord_roughness([60, 62]), fails)
    _expect_lt("C major < cluster", chord_roughness([60, 64, 67]),
                                    chord_roughness([60, 61, 62]), fails)
    _expect_lt("Cmaj7 < Cm7b5",     chord_roughness([60, 64, 67, 71]),
                                    chord_roughness([60, 63, 66, 70]), fails)

    print("\n=== Composite (Sethares + Stolzenburg) ordering ===")
    print("Adds the tonal checks that pure Sethares fails:")
    c_m3 = composite_dissonance([60, 64]).composite
    c_tt = composite_dissonance([60, 66]).composite
    _expect_lt("M3 < tritone (composite restores tonal intuition)", c_m3, c_tt, fails)
    c_cmaj = composite_dissonance([60, 64, 67]).composite
    c_cmin = composite_dissonance([60, 63, 67]).composite
    _expect_lt("C major < C minor (composite)", c_cmaj, c_cmin, fails)
    c_c7 = composite_dissonance([60, 64, 67, 70]).composite
    c_cdim7 = composite_dissonance([60, 63, 66, 69]).composite
    _expect_lt("C7 < Cdim7 (composite; dim chord's inharmonicity surfaces)",
               c_c7, c_cdim7, fails)

    if fails:
        print(f"\n{len(fails)} ordering check(s) failed: {fails}")
        return 1
    print("\nAll ordering checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
