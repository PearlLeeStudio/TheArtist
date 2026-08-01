"""D5: Paired significance test between every pair of methods.

For each ordered pair (method_a, method_b) of the 5 PEFT methods, we
treat each genre as a paired sample (mean top-1 across seeds is the
per-genre statistic). Then run a Wilcoxon signed-rank test across the
11 genres. The output is a triangular matrix of p-values and a CSV
with the full statistics.

Inputs:
    results/peft_compare_all.csv produced by eval_peft_compare.py
    (must contain columns: method, genre, seed, test_top1)

Outputs:
    results/method_pair_significance.csv — long-format table
    results/method_pair_significance.md  — triangular p-value matrix
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_DIR / "results" / "peft_compare_all.csv"
OUT_CSV = PROJECT_DIR / "results" / "method_pair_significance.csv"
OUT_MD = PROJECT_DIR / "results" / "method_pair_significance.md"

METHODS = ["full_ft", "lora", "bitfit", "ia3", "prefix"]


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input not found: {INPUT_CSV}. Run eval_peft_compare.py --all-genres first.")

    df = pd.read_csv(INPUT_CSV)
    # Normalize method column to our canonical set
    df["method"] = df["method"].replace({
        "Full FT": "full_ft", "full": "full_ft", "LoRA": "lora",
        "BitFit": "bitfit", "IA3": "ia3", "Prefix": "prefix",
        "prefix_tuning": "prefix",
    })

    # Per (method, genre) mean top1 across seeds
    if "seed" in df.columns:
        agg = df.groupby(["method", "genre"], as_index=False)["test_top1"].mean()
    else:
        agg = df[["method", "genre", "test_top1"]].copy()

    rows = []
    for a, b in combinations(METHODS, 2):
        va = agg[agg["method"] == a].set_index("genre")["test_top1"]
        vb = agg[agg["method"] == b].set_index("genre")["test_top1"]
        common = va.index.intersection(vb.index)
        if len(common) < 4:
            continue
        diff = (va.loc[common] - vb.loc[common]).values
        # Wilcoxon signed-rank (one-sample on the diffs vs 0)
        try:
            res = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            stat, pval = res.statistic, res.pvalue
        except ValueError:
            stat, pval = float("nan"), float("nan")
        mean_diff = float(np.mean(diff))
        n_a_better = int((diff > 0).sum())
        rows.append({
            "method_a": a, "method_b": b,
            "n_genres": int(len(common)),
            "mean_diff_top1": round(mean_diff, 3),
            "n_a_better": n_a_better,
            "n_b_better": int((diff < 0).sum()),
            "wilcoxon_statistic": round(stat, 3) if not np.isnan(stat) else float("nan"),
            "p_value": round(pval, 4) if not np.isnan(pval) else float("nan"),
            "significant_p05": (pval < 0.05) if not np.isnan(pval) else False,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} method-pair tests to {OUT_CSV}")

    with OUT_MD.open("w") as f:
        f.write("# Method-pair significance test (Wilcoxon signed-rank across 11 genres)\n\n")
        f.write("Each row: a > b if `mean_diff_top1` is positive; significance at p<0.05.\n\n")
        f.write(out_df.to_markdown(index=False))
    print(f"Wrote markdown summary to {OUT_MD}")


if __name__ == "__main__":
    main()
