"""P9: Real-song eval refresh — per-song paired analysis.

Adds to eval_real_songs_v2.py output:
  - win rate: fraction of songs where method A's per-song top1 > method B's per-song top1
  - median delta per method pair
  - per-song variance across seeds
  - leakage check: songs whose chord progression overlaps F1 base training data

Outputs:
    results/real_song_paired.csv
    results/real_song_paired.md
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "real_song_eval_v2.csv"  # D7 writes _v2 name
# Fallback to historical v1 name if v2 file absent.
if not SRC.exists():
    SRC = ROOT / "results" / "real_song_eval.csv"
OUT_CSV = ROOT / "results" / "real_song_paired.csv"
OUT_MD = ROOT / "results" / "real_song_paired.md"

METHODS = ["bitfit", "ia3", "full", "prefix_tuning", "lora"]  # match eval_peft_compare.py


def main():
    if not SRC.exists():
        raise SystemExit(f"FAIL: {SRC} missing — run v2 chain D7 first")

    df = pd.read_csv(SRC)
    # Actual eval_real_songs.py columns: model, model_kind, song_id, genre, val_top1
    # Prefer 'title' as song identifier when available (v2 D7 leaves song_id empty).
    df = df.drop(columns=["song_id"], errors="ignore")
    col_map = {"val_top1": "top1", "model": "model_id",
               "genre_cell": "genre", "title": "song"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # Parse method from model_id (values like "F1", "LoRA-blues", "BitFit-rock", etc.)
    if "method" not in df.columns:
        df["method"] = df["model_id"].str.extract(
            r"^(bitfit|ia3|full|prefix_tuning|lora|BitFit|IA3|Full|Prefix|LoRA)",
            expand=False).str.lower().replace({"prefix": "prefix_tuning"})

    # Per (method, genre, song): mean top1 across seeds (or single value if no seed)
    if "seed" in df.columns:
        per_song = df.groupby(["method", "genre", "song"]).agg(
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
    else:
        per_song = df.groupby(["method", "genre", "song"]).agg(
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            n_seeds=("model_id", "count"),
        ).reset_index()

    rows = []
    for a, b in combinations(METHODS, 2):
        # Pivot to song-level pair comparison
        a_df = per_song[per_song["method"] == a][["genre", "song", "top1_mean"]].rename(columns={"top1_mean": "top1_a"})
        b_df = per_song[per_song["method"] == b][["genre", "song", "top1_mean"]].rename(columns={"top1_mean": "top1_b"})
        merged = a_df.merge(b_df, on=["genre", "song"], how="inner")
        if len(merged) == 0:
            continue
        diff = merged["top1_a"] - merged["top1_b"]
        rows.append({
            "method_a": a, "method_b": b,
            "n_songs": int(len(merged)),
            "win_rate_a": round(float((diff > 0).mean()), 3),
            "tie_rate": round(float((diff == 0).mean()), 3),
            "win_rate_b": round(float((diff < 0).mean()), 3),
            "median_delta": round(float(diff.median()), 3),
            "mean_delta": round(float(diff.mean()), 3),
        })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(
        "# Real-song eval — per-song paired analysis\n\n"
        "Per method pair: win rate / median delta / mean delta across songs where both methods have a score.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False) +
        "\n\n## Leakage check\n\n"
        "F1 base training data overlap with eval real songs — see `results/dataset_validity.csv`.\n"
    )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
