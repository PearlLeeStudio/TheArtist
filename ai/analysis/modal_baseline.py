"""Measure natural-occurrence rate of modal interchange in current chord
generation, per genre, against the home key.

Generates N progressions per genre via the production inference path,
classifies each chord against the home key, and writes per-chord rows
plus an aggregate summary.

Usage:
    cd ai && uv run python -m analysis.modal_baseline --n 5 --bars 8
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

Classification = Literal[
    "DIATONIC", "MODAL_INTERCHANGE", "SECONDARY_DOMINANT",
    "TRITONE_SUB", "OTHER", "UNPARSEABLE",
]

# Speed: skip the slow LLM-judged R2 rerank, per-transition explanations,
# and voice leading. None of these change the candidate chord distribution.
os.environ.setdefault("THEARTIST_DISABLE_RAG", "1")
os.environ.setdefault("THEARTIST_DISABLE_EXPLANATIONS", "1")
os.environ.setdefault("THEARTIST_DISABLE_VOICE_LEADING", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.services.chord_generator import MAJOR_INTERVALS, MINOR_INTERVALS  # noqa: E402
from app.services.genre_tables import GENRE_DEFAULTS  # noqa: E402
from app.services.inference import generate_ai_suggestions  # noqa: E402
from app.services.song_form import NOTE_TO_SEMI, _parse_chord_root  # noqa: E402

# ── Modal interchange detector ─────────────────────────────────────────

MAJOR_DIATONIC = set(MAJOR_INTERVALS)         # I ii iii IV V vi vii°
MINOR_DIATONIC = set(MINOR_INTERVALS)         # i ii° bIII iv v bVI bVII (natural)

# Expected qualities at each diatonic degree. Multiple acceptable per slot
# because (a) V can be triad/dom7/sus, (b) ii° in minor can render as dim or
# half-diminished, (c) v in minor can be minor (natural) or dominant (harmonic).
DIATONIC_QUALITIES_MAJOR: dict[int, set[str]] = {
    0:  {"maj"},
    2:  {"min"},
    4:  {"min"},
    5:  {"maj"},
    7:  {"maj", "dom7", "sus"},
    9:  {"min"},
    11: {"dim", "half_dim"},
}
DIATONIC_QUALITIES_MINOR: dict[int, set[str]] = {
    0:  {"min"},
    2:  {"dim", "half_dim", "min"},
    3:  {"maj"},
    5:  {"min", "dom7"},
    7:  {"min", "maj", "dom7"},   # natural v / harmonic V / dominant V7
    8:  {"maj"},
    10: {"maj", "dom7"},
}

# Degree-name lookup for human-readable reporting
DEGREE_NAME_MAJOR = {
    0: "I", 1: "bII", 2: "ii", 3: "bIII", 4: "iii", 5: "IV",
    6: "#IV/bV", 7: "V", 8: "bVI", 9: "vi", 10: "bVII", 11: "vii°",
}
DEGREE_NAME_MINOR = {
    0: "i", 1: "bii", 2: "ii°", 3: "bIII", 4: "III", 5: "iv",
    6: "#iv/bv", 7: "v/V", 8: "bVI", 9: "VI", 10: "bVII", 11: "vii°",
}


def parse_key(key_str: str) -> tuple[int, str]:
    """'C major' → (0, 'major'); 'A minor' → (9, 'minor')."""
    parts = key_str.strip().split()
    return NOTE_TO_SEMI[parts[0]], parts[1].lower()


def chord_quality_class(quality: str) -> str:
    """Bucket a tokenizer quality string into a coarse class.

    Vocab reference (ai/training/tokenizer.py QUALITIES): maj, m, 7, maj7,
    m7, m7b5, dim7, dim, aug, sus4, sus2, 6, m6, 9, m9, maj9, 11, m11, 13,
    m13, add9, mMaj7, 7b9, 7#9, 7#11, 7b13.
    """
    q = quality.strip()
    if q in ("", "maj", "maj7", "maj9", "6", "add9"):
        return "maj"
    if q in ("sus4", "sus2"):
        return "sus"
    if q == "aug":
        return "aug"
    if q in ("dim", "dim7"):
        return "dim"
    if q == "m7b5":
        return "half_dim"
    if q == "mMaj7":
        return "min"
    if q.startswith("m"):  # m, m6, m7, m9, m11, m13
        return "min"
    if q in ("7", "9", "11", "13", "7b9", "7#9", "7#11", "7b13"):
        return "dom7"
    return "unknown"


STABLE_BORROW_QUALITIES = {"maj", "min", "dom7", "dim", "half_dim", "aug"}


def classify_chord(chord: str, key_root: int, mode: str) -> tuple[Classification, int | None, str]:
    """Classify a chord against the home key.

    Returns (classification, degree_from_tonic, quality_class). A chord is
    `MODAL_INTERCHANGE` when it has a stable triad/7 quality and is either
    chromatic or sits on a diatonic degree but with a non-diatonic quality
    (covers iv-in-major, Picardy I, non-dom7 secondaries / chromatic
    mediants like II maj / III maj). V7 → I is caught earlier as diatonic;
    secondary-dom and TT-sub get their own buckets when qc is dom7.
    """
    parsed = _parse_chord_root(chord)
    if parsed is None:
        return ("UNPARSEABLE", None, "?")
    root_str, qual = parsed
    deg = (NOTE_TO_SEMI[root_str] - key_root) % 12
    qc = chord_quality_class(qual)

    if mode == "major":
        home, home_q = MAJOR_DIATONIC, DIATONIC_QUALITIES_MAJOR
    else:
        home, home_q = MINOR_DIATONIC, DIATONIC_QUALITIES_MINOR

    if deg in home and qc in home_q.get(deg, set()):
        return ("DIATONIC", deg, qc)

    if qc == "dom7":
        target = (deg + 5) % 12
        if target in home and target != 0:
            return ("SECONDARY_DOMINANT", deg, qc)
        if (deg + 1) % 12 in home:
            return ("TRITONE_SUB", deg, qc)

    if qc in STABLE_BORROW_QUALITIES:
        return ("MODAL_INTERCHANGE", deg, qc)

    return ("OTHER", deg, qc)


# ── Generation harness ─────────────────────────────────────────────────


def measure(n_per_genre: int, n_bars: int) -> list[dict]:
    rows: list[dict] = []
    genres = list(GENRE_DEFAULTS.keys())

    for gi, genre in enumerate(genres, 1):
        defaults = GENRE_DEFAULTS[genre]
        key = defaults["key"]
        bpm = defaults["bpm"]
        ts = defaults["time_signature"]
        key_root, mode = parse_key(key)

        print(f"[{gi:>2}/{len(genres)}] {genre:<11} key={key:<10} ", end="", flush=True)

        for trial in range(n_per_genre):
            try:
                suggestions = generate_ai_suggestions(
                    key=key,
                    genre=genre,
                    context=[],
                    selected_measures=list(range(n_bars)),
                    time_signature=ts,
                    bpm=bpm,
                )
            except Exception as e:
                print(f"\n  trial {trial}: ERROR {type(e).__name__}: {e}")
                continue
            if not suggestions:
                print(f"\n  trial {trial}: no suggestions (model unavailable?)")
                continue

            for sug_i, sug in enumerate(suggestions):
                chord_seq: list[str] = []
                for m in range(n_bars):
                    chord_seq.extend(sug.chords.get(str(m), []))
                for pos, ch in enumerate(chord_seq):
                    cls, deg, qc = classify_chord(ch, key_root, mode)
                    rows.append({
                        "genre": genre,
                        "key": key,
                        "mode": mode,
                        "trial": trial,
                        "sug": sug_i,
                        "pos": pos,
                        "chord": ch,
                        "degree": deg,
                        "quality_class": qc,
                        "classification": cls,
                    })
            print(".", end="", flush=True)
        print()

    return rows


# ── Reporting ──────────────────────────────────────────────────────────


def summarize(rows: list[dict]) -> str:
    by_genre: dict[str, Counter] = defaultdict(Counter)
    by_genre_borrowed: dict[str, Counter] = defaultdict(Counter)
    progressions_with_mi: dict[str, set] = defaultdict(set)
    progressions_total: dict[str, set] = defaultdict(set)

    for r in rows:
        g = r["genre"]
        by_genre[g][r["classification"]] += 1
        if r["classification"] == "MODAL_INTERCHANGE":
            name = (DEGREE_NAME_MAJOR if r["mode"] == "major" else DEGREE_NAME_MINOR).get(r["degree"], "?")
            by_genre_borrowed[g][f"{name} ({r['quality_class']})"] += 1
            progressions_with_mi[g].add((r["trial"], r["sug"]))
        progressions_total[g].add((r["trial"], r["sug"]))

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 88)
    lines.append("Modal interchange — natural occurrence in current AI generation")
    lines.append("=" * 88)
    lines.append(
        f"{'Genre':<12} {'N':>5} {'DIA%':>6} {'MI%':>6} {'V/x%':>6} "
        f"{'TT%':>5} {'OTH%':>5} {'UNP%':>5} {'MI/prog':>8}"
    )
    lines.append("-" * 88)

    overall = Counter()
    overall_progs = 0
    overall_progs_with_mi = 0

    for g in sorted(by_genre):
        ctr = by_genre[g]
        n = sum(ctr.values())
        if n == 0:
            continue
        progs = len(progressions_total[g])
        mi_progs = len(progressions_with_mi[g])
        mi_rate = (mi_progs / progs * 100) if progs else 0.0
        lines.append(
            f"{g:<12} {n:>5} "
            f"{100 * ctr['DIATONIC'] / n:>5.1f} "
            f"{100 * ctr['MODAL_INTERCHANGE'] / n:>5.1f} "
            f"{100 * ctr['SECONDARY_DOMINANT'] / n:>5.1f} "
            f"{100 * ctr['TRITONE_SUB'] / n:>4.1f} "
            f"{100 * ctr['OTHER'] / n:>4.1f} "
            f"{100 * ctr['UNPARSEABLE'] / n:>4.1f} "
            f"{mi_rate:>7.1f}%"
        )
        overall.update(ctr)
        overall_progs += progs
        overall_progs_with_mi += mi_progs

    n_all = sum(overall.values())
    if n_all:
        lines.append("-" * 88)
        lines.append(
            f"{'OVERALL':<12} {n_all:>5} "
            f"{100 * overall['DIATONIC'] / n_all:>5.1f} "
            f"{100 * overall['MODAL_INTERCHANGE'] / n_all:>5.1f} "
            f"{100 * overall['SECONDARY_DOMINANT'] / n_all:>5.1f} "
            f"{100 * overall['TRITONE_SUB'] / n_all:>4.1f} "
            f"{100 * overall['OTHER'] / n_all:>4.1f} "
            f"{100 * overall['UNPARSEABLE'] / n_all:>4.1f} "
            f"{100 * overall_progs_with_mi / overall_progs:>7.1f}%"
        )

    lines.append("")
    lines.append("Top borrowed degrees per genre (chord-count, role-in-home-key):")
    lines.append("-" * 88)
    for g in sorted(by_genre_borrowed):
        top = by_genre_borrowed[g].most_common(5)
        if not top:
            lines.append(f"{g:<12} (none)")
            continue
        rendered = ", ".join(f"{name}×{cnt}" for name, cnt in top)
        lines.append(f"{g:<12} {rendered}")

    lines.append("")
    lines.append("Legend: DIA=diatonic, MI=modal interchange, V/x=secondary dominant,")
    lines.append("        TT=tritone sub, OTH=chromatic/other, UNP=unparseable.")
    lines.append("        MI/prog = % of generated progressions containing ≥1 borrowed chord.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="progressions per genre (each yields 3 candidates)")
    ap.add_argument("--bars", type=int, default=8, help="bars per progression")
    ap.add_argument("--out", type=Path, default=ROOT / "ai" / "results" / "modal_interchange_baseline.csv")
    args = ap.parse_args()

    rows = measure(args.n, args.bars)
    if not rows:
        print("No rows produced — check that AI checkpoints are available.")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    print(summary)
    summary_path = args.out.with_suffix(".txt")
    summary_path.write_text(summary + "\n")
    print(f"\nWrote {len(rows)} chord events → {args.out}")
    print(f"Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
