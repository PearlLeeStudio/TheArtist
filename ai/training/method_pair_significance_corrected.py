"""P2: Wilcoxon signed-rank with Holm/BH multiple-comparison correction.

Extends method_pair_significance.py by adding:
- Holm correction (family-wise error rate)
- Benjamini-Hochberg correction (false discovery rate)
- Effect size (Cliff's delta) per method pair

Outputs:
    results/method_pair_significance_corrected.csv
    results/method_pair_significance_corrected.md
"""
from __future__ import annotations
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "peft_compare_all.csv"
OUT_CSV = ROOT / "results" / "method_pair_significance_corrected.csv"
OUT_MD = ROOT / "results" / "method_pair_significance_corrected.md"

METHODS = ["bitfit", "ia3", "full", "prefix_tuning", "lora"]  # match eval_peft_compare.py
GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]


def cliffs_delta(x, y):
    """Effect size: probability that x > y minus probability that x < y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return (greater - less) / (nx * ny)


def holm_correction(pvals):
    """Holm-Bonferroni: sort ascending, multiply by (n-rank+1), enforce monotone."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(pvals[idx] * (n - rank), 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def bh_correction(pvals):
    """Benjamini-Hochberg: sort ascending, multiply by n/rank, enforce monotone (rev)."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    running_min = 1.0
    for rank_from_end in range(n):
        idx = order[n - 1 - rank_from_end]
        adj = min(pvals[idx] * n / (n - rank_from_end), 1.0)
        running_min = min(running_min, adj)
        adjusted[idx] = running_min
    return adjusted


def main():
    df = pd.read_csv(SRC)
    df["method"] = df["method"].str.lower()
    df["genre"] = df["genre"].str.lower()
    # Drop F1 baseline rows (no method we're comparing)
    df = df[df["method"].isin(["lora", "ia3", "bitfit", "prefix_tuning", "full"])]

    # Per (method, genre): mean test_top1 across seeds
    grouped = df.groupby(["method", "genre"])["test_top1"].mean().unstack()

    rows = []
    for a, b in combinations(METHODS, 2):
        if a not in grouped.index or b not in grouped.index:
            continue
        common = [g for g in GENRES if not (pd.isna(grouped.loc[a, g]) if g in grouped.columns else True)
                                       and not (pd.isna(grouped.loc[b, g]) if g in grouped.columns else True)]
        if len(common) < 2:
            continue
        diff = grouped.loc[a, common].values - grouped.loc[b, common].values
        try:
            res = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            stat, pval = res.statistic, res.pvalue
        except ValueError:
            stat, pval = float("nan"), float("nan")

        # Cliff's delta on the SAME common genres used in the Wilcoxon test
        # (matches statistical scope; previously used all seed×genre rows which
        # could mix in genres only one method completed → distorts effect size).
        a_vals = df[(df["method"] == a) & (df["genre"].isin(common))]["test_top1"].values
        b_vals = df[(df["method"] == b) & (df["genre"].isin(common))]["test_top1"].values
        cd = cliffs_delta(a_vals, b_vals)

        rows.append({
            "method_a": a, "method_b": b,
            "n_genres": int(len(common)),
            "mean_diff_top1": round(float(np.mean(diff)), 3),
            "n_a_better": int((diff > 0).sum()),
            "n_b_better": int((diff < 0).sum()),
            "n_ties": int((diff == 0).sum()),
            "cliffs_delta": round(cd, 3),
            "wilcoxon_stat": round(stat, 3) if not np.isnan(stat) else float("nan"),
            "p_raw": round(pval, 4) if not np.isnan(pval) else float("nan"),
        })

    # Apply Holm + BH correction across all pair tests
    raw_pvals = np.array([r["p_raw"] if not np.isnan(r["p_raw"]) else 1.0 for r in rows])
    holm = holm_correction(raw_pvals)
    bh = bh_correction(raw_pvals)
    for r, h, bh_val in zip(rows, holm, bh):
        r["p_holm"] = round(float(h), 4)
        r["p_bh"] = round(float(bh_val), 4)
        # Write as 1/0 int so pandas round-trips the column type correctly
        # (Python bool → CSV writes "True"/"False" string, reads back as object
        # not bool, breaking downstream filters like `df[df["sig"] == True]`).
        r["sig_holm_p05"] = int(h < 0.05)
        r["sig_bh_p05"] = int(bh_val < 0.05)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out_df)} pair tests)")

    with OUT_MD.open("w") as f:
        f.write("# Method-pair significance with multiple-comparison correction\n\n")
        f.write("- **Raw p**: Wilcoxon signed-rank across 11 genres on per-genre seed means.\n")
        f.write("- **p_holm**: Holm-Bonferroni step-down (controls family-wise error).\n")
        f.write("- **p_bh**: Benjamini-Hochberg (controls false discovery rate).\n")
        f.write("- **Cliff's delta**: non-parametric effect size; |δ|>0.474 = large.\n\n")
        f.write(out_df.to_markdown(index=False))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
