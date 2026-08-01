"""P4: Macro summary of peft_compare_all.csv.

Per-method per-genre: mean ± std across 3 seeds.
Per-method macro: mean of per-genre means (not pooled, to avoid large-genre dominance).
Win counts: how often each method has highest mean top1 per genre.

Outputs:
    results/macro_summary.csv   — long format
    results/macro_summary.md    — paper-ready table
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "peft_compare_all.csv"
OUT_CSV = ROOT / "results" / "macro_summary.csv"
OUT_MD = ROOT / "results" / "macro_summary.md"

METHODS = ["bitfit", "ia3", "full", "prefix_tuning", "lora"]  # match eval_peft_compare.py
GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]


def main():
    df = pd.read_csv(SRC)
    df["method"] = df["method"].str.lower()
    df["genre"] = df["genre"].str.lower()
    # eval_peft_compare.py writes test_top1/test_loss; drop F1 baseline rows
    # (which have method='F1 (no adapt)' and no seed) from method aggregation.
    df = df[df["method"].isin(["lora", "ia3", "bitfit", "prefix_tuning", "full"])]

    # Per (method, genre) mean and std across 3 seeds
    agg = df.groupby(["method", "genre"]).agg(
        mean_top1=("test_top1", "mean"),
        std_top1=("test_top1", "std"),
        mean_val_loss=("test_loss", "mean"),
        std_val_loss=("test_loss", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg.to_csv(OUT_CSV, index=False)

    # Per-method macro (mean + std of per-genre means across 11 genres).
    # macro_top1_std measures genre-level spread (how consistently a method
    # works across genres) — distinct from per-(method,genre) seed std above.
    macro = agg.groupby("method").agg(
        macro_top1=("mean_top1", "mean"),
        macro_top1_std=("mean_top1", "std"),
        macro_val_loss=("mean_val_loss", "mean"),
        macro_val_loss_std=("mean_val_loss", "std"),
    ).reset_index()

    # Win count: per genre, which method has the highest mean_top1
    wins = {m: 0 for m in METHODS}
    pivot = agg.pivot(index="genre", columns="method", values="mean_top1")
    for g in pivot.index:
        winner = pivot.loc[g].idxmax()
        if winner in wins:
            wins[winner] += 1

    # Markdown summary
    lines = ["# Macro summary across 11 genres × 3 seeds", ""]
    lines.append("## Per-method macro (mean of per-genre means)")
    lines.append("")
    lines.append("| Method | Macro Top-1 (mean ± std across genres) | Macro val_loss | Wins |")
    lines.append("|---|---:|---:|---:|")
    for _, row in macro.iterrows():
        m = row["method"]
        lines.append(f"| {m} | {row['macro_top1']:.2f} ± {row['macro_top1_std']:.2f} | "
                     f"{row['macro_val_loss']:.4f} ± {row['macro_val_loss_std']:.4f} | "
                     f"{wins.get(m, 0)}/11 |")
    lines.append("")

    lines.append("## Per-method per-genre (mean ± std across 3 seeds)")
    lines.append("")
    header = "| Genre | " + " | ".join(METHODS) + " |"
    sep = "|" + "---|" * (len(METHODS) + 1)
    lines.append(header)
    lines.append(sep)
    for g in GENRES:
        cells = [g]
        for m in METHODS:
            sub = agg[(agg["method"] == m) & (agg["genre"] == g)]
            if len(sub) == 0:
                cells.append("—")
            else:
                cells.append(f"{sub['mean_top1'].iloc[0]:.2f}±{sub['std_top1'].iloc[0]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
