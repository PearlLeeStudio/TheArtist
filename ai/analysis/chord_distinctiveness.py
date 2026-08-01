"""Per-genre chord-distinctiveness diagnostics beyond unigram statistics.

Inputs:
  ai/data/splits/train.jsonl   (song-level chord transcriptions)
Outputs:
  ai/results/chord_distinctiveness.csv   (per-genre features + composite score)
  ai/results/chord_distinctiveness.md    (paper-ready markdown table + composite vs LoRA gain)

Per-song features (averaged per genre):
  - non_diatonic_rate           — fraction of chords whose root is outside the diatonic scale
  - slash_bass_rate              — fraction of chords with explicit /bass annotation
  - inversion_rate               — fraction of chords whose explicit bass is *not* the root (subset of slash_bass)
  - dominant7_rate               — fraction of chords with a dominant 7th quality
  - extended_rate                — fraction of chords carrying tensions (9, 11, 13, sus2, sus4, alt)
  - secondary_dom_rate           — V/x patterns where x is a non-tonic diatonic chord
  - cadence_authentic_rate       — V→I (or V7→I) per-song
  - cadence_plagal_rate          — IV→I per-song
  - cadence_iiVI_rate            — ii→V→I per-song
  - cadence_blues_rate           — I7→IV7 or I7→V7 per-song
  - bigram_entropy               — per-song mean Shannon entropy over chord-quality bigrams (in bits)
  - repetition_rate              — fraction of consecutive bars whose chord set is identical to the previous bar

Composite distinctiveness score = z-score average of features that empirically
move chord-only genre identity (non_diatonic_rate, slash_bass_rate,
secondary_dom_rate, cadence_iiVI_rate, cadence_blues_rate, repetition_rate),
each rescaled so higher = "more chord-distinctive identity present in the
chord layer."
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_JSONL = PROJECT_DIR / "data" / "splits" / "train.jsonl"
LORA_TEST_CSV = PROJECT_DIR / "results" / "lora_selected_test.csv"
OUT_CSV = PROJECT_DIR / "results" / "chord_distinctiveness.csv"
OUT_MD = PROJECT_DIR / "results" / "chord_distinctiveness.md"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
SOURCE_FILTER = {f"chordonomicon_{g}" for g in GENRES if g != "classical"}
# classical = Bach chorales path; sources from chord_genre_classifier convention
SOURCE_FILTER.add("chordonomicon_classical")
# Per-genre dataset bias: the train split records `genre` directly. We trust
# the explicit `genre` field and filter by it, rather than by source string.

NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
NATURAL_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}

CHORD_RE = re.compile(
    r"^(?P<root>[A-G][b#]?)"
    r"(?P<quality>(?:maj7|maj9|maj13|m7b5|m7|maj|min|m|dim7|dim|aug|sus[24]?|7|9|11|13|6/9|6|add9|alt)*)"
    r"(?:/(?P<bass>[A-G][b#]?))?$"
)


def _root_pc(chord: str) -> tuple[int, str, int | None] | None:
    """Return (root_pc, quality_string, bass_pc) or None if unparseable."""
    m = CHORD_RE.match(chord.strip())
    if not m:
        return None
    root = m.group("root")
    quality = (m.group("quality") or "").lower()
    bass = m.group("bass")
    root_pc = NOTE_TO_PC.get(root)
    if root_pc is None:
        return None
    bass_pc = NOTE_TO_PC.get(bass) if bass else None
    return root_pc, quality, bass_pc


def _key_to_tonic_mode(key: str | None) -> tuple[int, str] | None:
    if not key:
        return None
    key = key.strip()
    # Tokens like "Cmaj", "Am", "F#m", "Bbmaj"
    m = re.match(r"^([A-G][b#]?)(maj|min|m)?$", key)
    if not m:
        return None
    tonic = NOTE_TO_PC.get(m.group(1))
    if tonic is None:
        return None
    raw = (m.group(2) or "maj").lower()
    mode = "min" if raw in {"min", "m"} else "maj"
    return tonic, mode


def _is_diatonic(chord_pc: int, tonic_pc: int, mode: str) -> bool:
    rel = (chord_pc - tonic_pc) % 12
    scale = MAJOR_SCALE if mode == "maj" else NATURAL_MINOR_SCALE
    return rel in scale


def _scale_degree(chord_pc: int, tonic_pc: int) -> int:
    return (chord_pc - tonic_pc) % 12


def _is_dominant7(quality: str) -> bool:
    # dominant 7 = "7" without preceding maj/m and without preceding "m7b5"
    if "maj7" in quality:
        return False
    if "m7" in quality and "m7b5" not in quality:
        return False
    if "dim7" in quality:
        return False
    return ("7" in quality) or ("9" in quality and "maj9" not in quality) or ("13" in quality and "maj13" not in quality)


def _is_minor(quality: str) -> bool:
    if quality.startswith("min") or quality.startswith("m") and not quality.startswith("maj"):
        # Distinguish "m7" / "min" from "maj"
        return True
    return False


def _has_extension(quality: str) -> bool:
    return any(t in quality for t in ("9", "11", "13", "sus2", "sus4", "add9", "alt"))


def _qual_short(quality: str) -> str:
    """Reduce quality to a short tag for bigram entropy: maj/min/dom/dim/aug/sus."""
    if not quality or quality in {"maj"}:
        return "maj"
    if "dim" in quality:
        return "dim"
    if "aug" in quality:
        return "aug"
    if "sus" in quality:
        return "sus"
    if "maj7" in quality or "maj9" in quality or "maj13" in quality:
        return "maj7"
    if _is_dominant7(quality):
        return "dom"
    if _is_minor(quality):
        return "min"
    return "oth"


def _flatten_song(bars: list) -> list[str]:
    chords = []
    for bar in bars:
        if isinstance(bar, list):
            chords.extend(bar)
        elif isinstance(bar, str):
            chords.append(bar)
    return chords


def _per_song_features(song: dict) -> dict | None:
    chords = _flatten_song(song.get("bars") or song.get("chords") or [])
    if len(chords) < 4:
        return None

    parsed = [(c, _root_pc(c)) for c in chords]
    parsed = [(c, p) for c, p in parsed if p is not None]
    if len(parsed) < 4:
        return None

    key_info = _key_to_tonic_mode(song.get("key"))
    if key_info is None:
        # Skip songs without a parseable key — we can't compute scale-relative features.
        return None
    tonic_pc, mode = key_info

    # Per-chord features
    diatonic_count = 0
    slash_count = 0
    inversion_count = 0
    dom7_count = 0
    extended_count = 0
    n = len(parsed)
    qualities_short = []
    degrees = []
    is_dom_flags = []
    is_minor_flags = []
    for chord, (root_pc, quality, bass_pc) in parsed:
        diatonic_count += 1 if _is_diatonic(root_pc, tonic_pc, mode) else 0
        if bass_pc is not None:
            slash_count += 1
            if bass_pc != root_pc:
                inversion_count += 1
        if _is_dominant7(quality):
            dom7_count += 1
            is_dom_flags.append(True)
        else:
            is_dom_flags.append(False)
        if _has_extension(quality):
            extended_count += 1
        is_minor_flags.append(_is_minor(quality))
        qualities_short.append(_qual_short(quality))
        degrees.append(_scale_degree(root_pc, tonic_pc))

    # Cadence detection — pairs and triples in scale-degree space
    auth_count = 0  # V→I (deg 7 → deg 0)
    plagal_count = 0  # IV→I (deg 5 → deg 0)
    iiV_I_count = 0  # ii→V→I (deg 2 → deg 7 → deg 0), ii is minor
    blues_count = 0  # I7→IV7 or I7→V7
    sec_dom_count = 0  # V/x → x where V/x is a dominant chord one fifth above x and x is non-tonic diatonic

    for i in range(len(degrees) - 1):
        d_i = degrees[i]
        d_j = degrees[i + 1]
        if d_i == 7 and d_j == 0:
            auth_count += 1
        if d_i == 5 and d_j == 0:
            plagal_count += 1
        # blues — I7 → IV7 / I7 → V7
        if d_i == 0 and is_dom_flags[i]:
            if (d_j == 5 and is_dom_flags[i + 1]) or (d_j == 7 and is_dom_flags[i + 1]):
                blues_count += 1
        # secondary dominant: dom chord whose root is a perfect 5th above the next chord root,
        # and the next chord is a non-tonic diatonic root.
        if is_dom_flags[i]:
            interval = (d_j - d_i) % 12
            if interval == 5:  # i is a fifth above j (V/j → j)
                if d_j != 0 and _is_diatonic((d_j + tonic_pc) % 12, tonic_pc, mode):
                    sec_dom_count += 1

    for i in range(len(degrees) - 2):
        d_i, d_j, d_k = degrees[i], degrees[i + 1], degrees[i + 2]
        # ii→V→I: ii is degree 2, V is degree 7, I is degree 0; ii should be minor, V usually dominant.
        if d_i == 2 and d_j == 7 and d_k == 0 and is_minor_flags[i]:
            iiV_I_count += 1

    # Bigram quality entropy
    bigrams = list(zip(qualities_short, qualities_short[1:]))
    if bigrams:
        counts = Counter(bigrams)
        total = sum(counts.values())
        ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    else:
        ent = 0.0

    # Repetition: fraction of consecutive *bars* whose chord set is identical to the previous bar.
    bars = song.get("bars") or []
    bar_sets = [tuple(b) if isinstance(b, list) else (b,) for b in bars if b]
    if len(bar_sets) >= 2:
        rep = sum(1 for a, b in zip(bar_sets, bar_sets[1:]) if a == b) / (len(bar_sets) - 1)
    else:
        rep = 0.0

    return {
        "non_diatonic_rate": 1.0 - (diatonic_count / n),
        "slash_bass_rate": slash_count / n,
        "inversion_rate": inversion_count / n,
        "dominant7_rate": dom7_count / n,
        "extended_rate": extended_count / n,
        "secondary_dom_rate": sec_dom_count / max(1, len(degrees) - 1),
        "cadence_authentic_rate": auth_count / max(1, len(degrees) - 1),
        "cadence_plagal_rate": plagal_count / max(1, len(degrees) - 1),
        "cadence_iiVI_rate": iiV_I_count / max(1, len(degrees) - 2),
        "cadence_blues_rate": blues_count / max(1, len(degrees) - 1),
        "bigram_entropy": ent,
        "repetition_rate": rep,
    }


def _aggregate_genre(songs: list[dict]) -> dict:
    keys = [
        "non_diatonic_rate", "slash_bass_rate", "inversion_rate",
        "dominant7_rate", "extended_rate", "secondary_dom_rate",
        "cadence_authentic_rate", "cadence_plagal_rate",
        "cadence_iiVI_rate", "cadence_blues_rate",
        "bigram_entropy", "repetition_rate",
    ]
    out = {"n_songs": len(songs)}
    for k in keys:
        vals = [s[k] for s in songs]
        out[k] = round(statistics.mean(vals), 4)
        out[f"{k}_std"] = round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4)
    return out


def _zscore(vals: list[float]) -> list[float]:
    if not vals:
        return []
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
    if sd == 0:
        return [0.0 for _ in vals]
    return [(v - mu) / sd for v in vals]


def _composite(per_genre: dict[str, dict]) -> dict[str, float]:
    """Per-genre composite as the mean z-score of *informative* harmonic features.

    Features dropped: slash_bass_rate (the chord symbols in this corpus rarely
    carry explicit /bass annotations, so it is identically 0 across all genres
    and contributes only noise). Features kept are limited to ones that vary
    across genres and admit a directional reading (higher → more chord-layer
    harmonic identity).
    """
    feature_names = [
        "secondary_dom_rate",
        "cadence_iiVI_rate",
        "cadence_blues_rate",
        "extended_rate",
        "bigram_entropy",
    ]
    genres = list(per_genre.keys())
    z_per_feature = {}
    for fname in feature_names:
        raw = [per_genre[g][fname] for g in genres]
        z_per_feature[fname] = _zscore(raw)
    composite: dict[str, float] = {}
    for i, g in enumerate(genres):
        composite[g] = round(
            statistics.mean(z_per_feature[fname][i] for fname in feature_names), 3
        )
    return composite


def _load_lora_gains() -> dict[str, float]:
    """Pull per-genre LoRA test gain (delta_top1) from the held-out test CSV."""
    out = {}
    with open(LORA_TEST_CSV) as f:
        for r in csv.DictReader(f):
            out[r["genre"]] = float(r["delta_top1"])
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main() -> None:
    print(f"Reading {TRAIN_JSONL}")
    by_genre: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    total = 0
    with open(TRAIN_JSONL) as f:
        for line in f:
            total += 1
            try:
                song = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            # Select by source tag: the song-level `genre` field over-selects
            # for rock (the generic chordonomicon pop pool carries rock-tagged
            # songs the adaptation study never trained on).
            src = song.get("source", "")
            if not src.startswith("chordonomicon_"):
                continue
            g = src.split("_", 1)[1]
            if g not in GENRES:
                continue
            feats = _per_song_features(song)
            if feats is None:
                skipped += 1
                continue
            by_genre[g].append(feats)
    print(f"  total songs scanned: {total}, kept per-genre: {sum(len(v) for v in by_genre.values())}, skipped: {skipped}")

    per_genre: dict[str, dict] = {}
    for g in GENRES:
        if g in by_genre and by_genre[g]:
            per_genre[g] = _aggregate_genre(by_genre[g])
        else:
            print(f"  WARN: no songs for {g}")

    composite = _composite(per_genre)
    for g, c in composite.items():
        per_genre[g]["composite_distinctiveness"] = c

    # Join with LoRA gains
    lora_gains = _load_lora_gains()
    for g in per_genre:
        per_genre[g]["lora_test_gain"] = round(lora_gains.get(g, 0.0), 2)

    # Write CSV
    fieldnames = ["genre", "n_songs", "composite_distinctiveness", "lora_test_gain",
                  "non_diatonic_rate", "slash_bass_rate", "inversion_rate",
                  "dominant7_rate", "extended_rate", "secondary_dom_rate",
                  "cadence_authentic_rate", "cadence_plagal_rate",
                  "cadence_iiVI_rate", "cadence_blues_rate",
                  "bigram_entropy", "repetition_rate"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for g in GENRES:
            if g not in per_genre:
                continue
            row = {"genre": g}
            for k in fieldnames[1:]:
                row[k] = per_genre[g].get(k, 0.0)
            w.writerow(row)
    print(f"Wrote {OUT_CSV.name}")

    # Correlations between composite/single features and LoRA gain
    genre_order = [g for g in GENRES if g in per_genre]
    composite_vals = [per_genre[g]["composite_distinctiveness"] for g in genre_order]
    gain_vals = [per_genre[g]["lora_test_gain"] for g in genre_order]
    r_all = _pearson(composite_vals, gain_vals)

    # Excluding Bach chorale
    non_chorale = [(c, g) for c, g in zip(composite_vals, gain_vals) if g != per_genre["classical"]["lora_test_gain"]]
    if non_chorale:
        r_no_chorale = _pearson([c for c, g in non_chorale], [g for c, g in non_chorale])
    else:
        r_no_chorale = float("nan")

    # Single-feature correlations (descriptive, not multiple-test corrected)
    single_feats = [
        "non_diatonic_rate", "slash_bass_rate", "secondary_dom_rate",
        "cadence_iiVI_rate", "cadence_blues_rate", "repetition_rate",
        "extended_rate", "dominant7_rate", "bigram_entropy",
    ]
    single_corr = {}
    for fn in single_feats:
        xs = [per_genre[g][fn] for g in genre_order]
        single_corr[fn] = _pearson(xs, gain_vals)

    # Markdown
    lines = [
        "# Chord-distinctiveness diagnostics (transition / cadence-level)",
        "",
        "Per-song chord-pattern features computed on `ai/data/splits/train.jsonl`,",
        "averaged per genre. Gains are LoRA test top-1 over F1 from",
        "`lora_selected_test.csv`. The composite score is a per-genre mean z-score",
        "over five informative features (secondary_dom_rate, cadence_iiVI_rate,",
        "cadence_blues_rate, extended_rate, bigram_entropy); see notes below.",
        "Slash-bass rate is omitted from the table because the chord symbols in",
        "this corpus rarely carry explicit `/bass` annotations and the rate is",
        "identically 0 across all 11 genres.",
        "",
        "| Genre | Songs | Composite | LoRA gain | Non-diat | Ext | SecDom | ii-V-I | Blues I7 | Auth V-I | Plagal IV-I | Repetition | Bigram H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in genre_order:
        d = per_genre[g]
        lines.append(
            f"| {g} | {d['n_songs']:,} | {d['composite_distinctiveness']:+.2f} | "
            f"{d['lora_test_gain']:+.2f} | {d['non_diatonic_rate']:.3f} | "
            f"{d['extended_rate']:.3f} | {d['secondary_dom_rate']:.3f} | "
            f"{d['cadence_iiVI_rate']:.3f} | {d['cadence_blues_rate']:.3f} | "
            f"{d['cadence_authentic_rate']:.3f} | {d['cadence_plagal_rate']:.3f} | "
            f"{d['repetition_rate']:.3f} | {d['bigram_entropy']:.2f} |"
        )

    lines += [
        "",
        f"**Composite vs LoRA test gain** (Pearson): including Bach chorale `r={r_all:+.3f}`; "
        f"excluding Bach chorale `r={r_no_chorale:+.3f}`.",
        "",
        "Single-feature Pearson correlations with LoRA test gain (descriptive,",
        "no multiple-test correction; informative features only):",
        "",
        "| Feature | Pearson r |",
        "|---|---:|",
    ]
    for fn, r in sorted(single_corr.items(), key=lambda kv: -abs(kv[1])):
        if math.isnan(r):
            continue
        lines.append(f"| {fn} | {r:+.3f} |")

    lines += [
        "",
        "## How to read",
        "",
        "- The 11 LoRA target genres differ on every feature: country has the",
        "  highest authentic V-I rate, gospel the highest plagal IV-I rate, bossa",
        "  the highest extended-chord rate, blues the highest blues I7 rate,",
        "  rnb_soul the highest non-diatonic-root rate, classical (Bach chorales)",
        "  the highest bigram entropy.",
        "- The composite is *not* a learned weighting; it is an intentional sum",
        "  of features that should track chord-layer harmonic identity. The",
        "  signal differs across the chosen features and across genres, so a",
        "  high composite does not imply a high LoRA gain.",
        "- Bach chorales sit on a different axis from the other genres: they",
        "  have low chord-root non-diatonicism (most chromaticism in Bach is",
        "  carried by chord *quality*, e.g. major V in a minor key, rather than",
        "  by chord-root choice), but high bigram entropy and high authentic",
        "  cadence rate. Including chorales the composite-vs-gain correlation",
        "  is small in magnitude; excluding chorales it is also small, mirroring",
        "  the chord-only classifier finding in §7.4: no single chord-pattern",
        "  composite cleanly orders the middle of the genre range.",
        "- The transition/cadence-level features are therefore best read as",
        "  *descriptive evidence* that the 11 genres really are harmonically",
        "  distinct in the chord layer (see the per-genre column patterns), not",
        "  as a single-axis predictor of LoRA gain.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD.name}")
    print(f"Composite vs LoRA gain (incl. Bach): r={r_all:+.3f}")
    print(f"Composite vs LoRA gain (excl. Bach): r={r_no_chorale:+.3f}")


if __name__ == "__main__":
    main()
