"""P6: Per-genre dataset statistics for the §4 Data section.

For each genre and split (train/val/test):
  - n_sequences
  - mean/median/p5/p95 sequence length (in chord tokens, pre-padding)
  - unique chord vocabulary count
  - Shannon entropy of chord-token distribution
  - top-10 chord coverage (fraction of tokens in top-10 most-frequent chords)

Outputs:
    results/per_genre_stats.csv
    results/per_genre_stats.md
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
OUT_CSV = ROOT / "results" / "per_genre_stats.csv"
OUT_MD = ROOT / "results" / "per_genre_stats.md"

GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
SPLITS = ["train", "val", "test"]


def shannon_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def top_k_coverage(counts, k=10):
    total = sum(counts)
    if total == 0:
        return 0.0
    sorted_counts = sorted(counts, reverse=True)
    return sum(sorted_counts[:k]) / total


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = int(round((p / 100) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def main():
    rows = []
    for genre in GENRES:
        # Select by the dataset.py source tag (chordonomicon_<genre>, incl.
        # classical). The song-level `genre` field over-selects for rock: the
        # generic chordonomicon pop pool also carries rock-tagged songs that
        # the adaptation study never trained on (tag 152,509 vs source 39,128).
        for split in SPLITS:
            split_file = SPLIT_DIR / f"{split}.jsonl"
            if not split_file.exists():
                rows.append({"genre": genre, "split": split, "n_sequences": 0,
                             "note": "split file missing"})
                continue

            lengths = []
            chord_counter = Counter()
            n_seq = 0
            for line in split_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    song = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if song.get("source") != f"chordonomicon_{genre}":
                    continue
                bars = song.get("bars", [])
                # Each bar has chords; flatten
                tokens = [c for bar in bars for c in (bar if isinstance(bar, list) else [bar])]
                if not tokens:
                    continue
                n_seq += 1
                lengths.append(len(tokens))
                chord_counter.update(tokens)

            sorted_lens = sorted(lengths)
            rows.append({
                "genre": genre,
                "split": split,
                "n_sequences": n_seq,
                "mean_len": round(sum(lengths) / n_seq, 1) if n_seq else 0,
                "median_len": percentile(sorted_lens, 50),
                "p5_len": percentile(sorted_lens, 5),
                "p95_len": percentile(sorted_lens, 95),
                "unique_chords": len(chord_counter),
                "entropy_bits": round(shannon_entropy(list(chord_counter.values())), 2),
                "top10_coverage": round(top_k_coverage(list(chord_counter.values()), 10), 3),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")

    # Train-split table for paper
    train = df[df["split"] == "train"].drop(columns=["split"])
    OUT_MD.write_text(
        "# Per-genre dataset statistics (train split)\n\n" +
        train.to_markdown(index=False) +
        "\n\n## Per-split breakdown\n\n" +
        df.to_markdown(index=False)
    )
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
