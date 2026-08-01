"""Dataset duplicate / leakage diagnostic across train/val/test per genre.

Checks two duplicate rates per genre × split-pair:
  1. Exact chord-sequence duplicate rate (sequences identical after
     joining all chord tokens)
  2. Key-normalized duplicate rate (transpose every sequence to C major
     equivalent, then check for duplicates)

Output: ai/results/dataset_duplicates.csv with columns:
  genre, split_a, split_b, n_a, n_b, exact_dup_rate, key_norm_dup_rate

A high duplicate rate between train and test means the per-genre top-1
numbers are inflated (the test set is partly memorized). Reviewer-risk
mitigation.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
RESULTS_DIR = PROJECT_DIR / "results"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]

# The two headline corpora (jazz, pop) are NOT chordonomicon sub-genres; they
# are identified by `source` group, not by the `genre` field above. Cover them
# with a dedicated source-level pass so the leakage diagnostic also reports the
# jazz/pop train/test cross-split duplicate counts. Source groups mirror the
# authoritative definitions in training/dataset.py.
POP_SOURCES = {"chordonomicon", "billboard"}
JAZZ_SOURCES = {
    "jht", "wjazzd", "jaah", "jazzstandards",
    "choco:real-book", "choco:jazz-corpus", "bunks",
}
# label -> set of sources; rows are emitted with `genre` = the label.
SOURCE_GROUPS = {
    "jazz": JAZZ_SOURCES,
    "pop": POP_SOURCES,
}

# 12 chromatic pitch classes for key normalization
PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
INDEX_TO_PC = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _parse_root(chord: str) -> tuple[int, str] | None:
    """Return (pitch_class_index, quality_suffix) or None if not parseable."""
    if not chord or chord.startswith(("[", "<")):
        return None
    # Try 2-char root first (C#, Db, Eb, etc.)
    for n in (2, 1):
        if len(chord) >= n:
            root = chord[:n]
            if root in PITCH_CLASS:
                return PITCH_CLASS[root], chord[n:]
    return None


def _normalize_to_C(chords: list[str], song_key_root: str | None) -> tuple[str, ...]:
    """Transpose all chords so the song's key root becomes C. If song key
    is unknown, transpose so the first parseable chord becomes C."""
    parsed = [(c, _parse_root(c)) for c in chords]
    parsed_with_pc = [(orig, p) for orig, p in parsed if p is not None]
    if not parsed_with_pc:
        return tuple(chords)
    # Anchor pitch class
    if song_key_root and song_key_root in PITCH_CLASS:
        anchor_pc = PITCH_CLASS[song_key_root]
    else:
        anchor_pc = parsed_with_pc[0][1][0]
    delta = (-anchor_pc) % 12
    out = []
    for orig, parse in parsed:
        if parse is None:
            out.append(orig)
        else:
            pc, suffix = parse
            new_pc = (pc + delta) % 12
            out.append(INDEX_TO_PC[new_pc] + suffix)
    return tuple(out)


def _flatten_song(song: dict) -> list[str]:
    """Flatten a song's bars into a chord-only sequence."""
    chords: list[str] = []
    for bar in song.get("bars", []):
        for c in bar:
            if c and not c.startswith(("[", "<")):
                chords.append(c)
    return chords


def _load_split_by_genre(split: str) -> dict[str, list[dict]]:
    """Returns {genre: [songs]}."""
    by_genre: dict[str, list[dict]] = defaultdict(list)
    path = SPLITS_DIR / f"{split}.jsonl"
    if not path.exists():
        return by_genre
    with open(path) as f:
        for line in f:
            song = json.loads(line)
            # Select by source tag: the `genre` field over-selects for rock
            # (the generic chordonomicon pop pool carries rock-tagged songs
            # outside the adaptation study's training subset).
            src = song.get("source", "")
            if not src.startswith("chordonomicon_"):
                continue
            genre = src.split("_", 1)[1]
            if genre in GENRES:
                by_genre[genre].append(song)
    return by_genre


def _load_split_by_source_group(split: str) -> dict[str, list[dict]]:
    """Returns {group_label: [songs]} for the headline source groups
    (jazz, pop) keyed by the `source` field rather than `genre`."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    path = SPLITS_DIR / f"{split}.jsonl"
    if not path.exists():
        return by_group
    with open(path) as f:
        for line in f:
            song = json.loads(line)
            source = song.get("source")
            for label, sources in SOURCE_GROUPS.items():
                if source in sources:
                    by_group[label].append(song)
                    break
    return by_group


def _dup_rate(set_a: set, set_b: set) -> float:
    """Fraction of set_a items also in set_b."""
    if not set_a:
        return 0.0
    return sum(1 for x in set_a if x in set_b) / len(set_a)


def _emit_rows(label: str, splits: dict[str, dict[str, list[dict]]], rows: list[dict]) -> None:
    """Compute per-split unique sets for `label` and append the three
    cross-split duplicate-rate rows. `splits` maps split_name -> {label: [songs]}."""
    per_split_exact = {}
    per_split_keynorm = {}
    for split_name, by_label in splits.items():
        songs = by_label.get(label, [])
        exact_set = {tuple(_flatten_song(s)) for s in songs}
        keynorm_set = {_normalize_to_C(_flatten_song(s), s.get("key_root")) for s in songs}
        per_split_exact[split_name] = exact_set
        per_split_keynorm[split_name] = keynorm_set
        print(f"  {label:12s} {split_name:5s}: {len(songs)} songs ({len(exact_set)} unique exact, {len(keynorm_set)} unique key-norm)")

    # Pair-wise duplicate rates: a-in-b for a ∈ {train, val, test}, b ∈ others
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n_a = len(per_split_exact[a])
        n_b = len(per_split_exact[b])
        exact_dup = _dup_rate(per_split_exact[a], per_split_exact[b])
        keynorm_dup = _dup_rate(per_split_keynorm[a], per_split_keynorm[b])
        rows.append({
            "genre": label,
            "split_a": a,
            "split_b": b,
            "n_a": n_a,
            "n_b": n_b,
            "exact_dup_rate_a_in_b": round(exact_dup, 4),
            "key_norm_dup_rate_a_in_b": round(keynorm_dup, 4),
        })


def main() -> None:
    splits = {s: _load_split_by_genre(s) for s in ("train", "val", "test")}
    # Headline corpora (jazz, pop) keyed by `source` group, not `genre`.
    source_splits = {s: _load_split_by_source_group(s) for s in ("train", "val", "test")}

    rows = []
    for genre in GENRES:
        _emit_rows(genre, splits, rows)
    # jazz / pop source-group pass (the two headline corpora)
    for label in SOURCE_GROUPS:
        _emit_rows(label, source_splits, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "dataset_duplicates.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Markdown summary
    out_md = RESULTS_DIR / "dataset_duplicates.md"
    with open(out_md, "w") as f:
        f.write("# Dataset duplicate / leakage diagnostic\n\n")
        f.write("Per-genre exact + key-normalized duplicate rate across splits.\n\n")
        f.write("| Genre | Pair | n_a | n_b | Exact a-in-b | Key-norm a-in-b |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['genre']} | {r['split_a']} → {r['split_b']} | {r['n_a']} | {r['n_b']} | {r['exact_dup_rate_a_in_b']:.2%} | {r['key_norm_dup_rate_a_in_b']:.2%} |\n")
        f.write("\n**Interpretation:**\n")
        f.write("- Exact duplicate rate > 1% in train→test → potential memorization risk.\n")
        f.write("- Key-norm duplicate rate captures cross-key matches (the same progression in different keys).\n")
        f.write("- High key-norm but low exact = augmentation overlap (expected for 12-key augmentation if applied pre-split).\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
