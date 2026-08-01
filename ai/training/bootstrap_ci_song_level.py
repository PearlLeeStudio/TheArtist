"""P3: Song-level / block bootstrap CI (replaces token-level D6).

Token-level CIs are too narrow because tokens within a song are correlated.
This script resamples at the SONG level instead: for each cell, take per-song
top-1 accuracy from cached evaluation, then bootstrap-resample songs.

Outputs:
    results/bootstrap_ci_song_level.csv
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PER_SONG = ROOT / "results" / "real_song_eval_v2.csv"  # D7 writes this name (v2 chain)
# Fallback to historical v1 name if v2 file absent.
if not PER_SONG.exists():
    PER_SONG = ROOT / "results" / "real_song_eval.csv"
OUT = ROOT / "results" / "bootstrap_ci_song_level.csv"

N_BOOT = 1000
RNG_SEED = 42


def main():
    if not PER_SONG.exists():
        raise SystemExit(f"FAIL: {PER_SONG} missing — run v2 chain step D7 first")

    df = pd.read_csv(PER_SONG)
    # Actual eval_real_songs.py columns: model, model_kind, song_id, genre, val_top1, val_top5, val_loss
    # Map to expected: model→model_id, song_id→song, val_top1→top1
    col_map = {"song_id": "song", "val_top1": "top1", "model": "model_id",
               "method": "model_id", "genre_cell": "genre", "title": "song"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "song" not in df.columns or "top1" not in df.columns:
        raise SystemExit("FAIL: real_song_eval.csv missing 'song_id' or 'val_top1' columns")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (model_id, genre), sub in df.groupby(["model_id", "genre"]):
        per_song_top1 = sub["top1"].values
        n_songs = len(per_song_top1)
        if n_songs == 0:
            continue
        # Bootstrap N_BOOT times
        boot_means = []
        for _ in range(N_BOOT):
            sample = rng.choice(per_song_top1, size=n_songs, replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)
        rows.append({
            "model_id": model_id,
            "genre": genre,
            "n_songs": n_songs,
            "point_top1": round(float(per_song_top1.mean()), 4),
            "ci_lower_95": round(float(np.percentile(boot_means, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(boot_means, 97.5)), 4),
            "ci_width": round(float(np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5)), 4),
        })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(rows)} cells × {N_BOOT} bootstrap resamples)")


if __name__ == "__main__":
    main()
