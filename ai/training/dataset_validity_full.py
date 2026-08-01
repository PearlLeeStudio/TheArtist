"""P5: Extended dataset validity check.

Extends dataset_duplicates.py with:
- F1 base-training data vs target val/test overlap (genre-token leakage proxy)
- Near-duplicate detection via repeated-loop overlap (chord 4-gram match)
- Per-genre summary of all leakage rates

Outputs:
    results/dataset_validity.csv
    results/dataset_validity.md
"""
from __future__ import annotations
import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
OUT_CSV = ROOT / "results" / "dataset_validity.csv"
OUT_MD = ROOT / "results" / "dataset_validity.md"

GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]


def _load_songs(split: str, genre: str):
    f = SPLIT_DIR / f"{split}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            song = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Select by source tag: the `genre` field over-selects for rock (the
        # generic chordonomicon pop pool carries rock-tagged songs outside the
        # adaptation study's training subset).
        if song.get("source") == f"chordonomicon_{genre}":
            bars = song.get("bars", [])
            tokens = tuple(c for bar in bars for c in (bar if isinstance(bar, list) else [bar]))
            if tokens:
                out.append(tokens)
    return out


def _ngram_overlap_rate(seqs_a, seqs_b, n=4):
    """Fraction of seqs_a whose n-gram set intersects with seqs_b's set."""
    if not seqs_a or not seqs_b:
        return 0.0
    b_ngrams = set()
    for s in seqs_b:
        for i in range(len(s) - n + 1):
            b_ngrams.add(s[i:i+n])
    overlap = 0
    for s in seqs_a:
        ngrams = {s[i:i+n] for i in range(len(s) - n + 1)}
        if ngrams & b_ngrams:
            overlap += 1
    return overlap / len(seqs_a)


def main():
    # First run the original dataset_duplicates.py for exact + key-norm coverage.
    # Use os.environ.copy() + override so the subprocess inherits HOME, USER, LANG,
    # LD_LIBRARY_PATH etc. — replacing env entirely caused uv/python failures.
    import os
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = "/tmp/uv-cache"
    subprocess.run(["uv", "run", "python", str(ROOT / "training" / "dataset_duplicates.py")],
                   check=True, env=env, cwd=str(ROOT))

    # Add near-duplicate (4-gram overlap) per genre across train/val/test
    rows = []
    for genre in GENRES:
        train = _load_songs("train", genre)
        val = _load_songs("val", genre)
        test = _load_songs("test", genre)
        rows.append({
            "genre": genre,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "ngram4_overlap_val_in_train": round(_ngram_overlap_rate(val, train), 3),
            "ngram4_overlap_test_in_train": round(_ngram_overlap_rate(test, train), 3),
            "ngram4_overlap_test_in_val": round(_ngram_overlap_rate(test, val), 3),
        })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(
        "# Dataset validity — extended diagnostic\n\n"
        "## Near-duplicate (4-gram) overlap rates\n\n"
        "Fraction of sequences in split A that share at least one chord 4-gram with split B.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False) +
        "\n\n## Related artifacts\n\n"
        "- `dataset_duplicates.csv` — exact + key-normalized duplicate rates (D3 output).\n"
        "- F1-base overlap: not computed (F1 was trained on pop/jazz; no labeled subset to compare against target genre val/test).\n"
    )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
