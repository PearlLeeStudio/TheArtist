"""Aggregate eval_results.csv from all experiments into unified CSVs.

Usage:
    cd ai
    uv run python training/aggregate_results.py
"""

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_DIR / "checkpoints"
OUT_DIR = PROJECT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    "phase0_pop_baseline",
    "ft_jazz_pop80",
    "ft_jazz_pop67",
    "ft_jazz_pop50",
    "ft_jazz_pop29",
    "ft_jazz_only",
]


def main():
    all_rows = []
    best_rows = []

    for exp in EXPERIMENTS:
        csv_path = CKPT_DIR / exp / "eval_results.csv"
        if not csv_path.exists():
            print(f"SKIP {exp}: no eval_results.csv")
            continue

        with open(csv_path) as f:
            reader = list(csv.DictReader(f))

        for row in reader:
            row["experiment"] = exp
            all_rows.append(row)

        # best = lowest val_loss
        best = min(reader, key=lambda r: float(r["val_loss"]))
        best["experiment"] = exp
        best_rows.append(best)

    # All epochs CSV
    if all_rows:
        fields = ["experiment"] + [k for k in all_rows[0] if k != "experiment"]
        with open(OUT_DIR / "all_experiments.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Saved {len(all_rows)} rows to results/all_experiments.csv")

    # Best per experiment CSV
    if best_rows:
        fields = ["experiment"] + [k for k in best_rows[0] if k != "experiment"]
        with open(OUT_DIR / "best_per_experiment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(best_rows)
        print(f"Saved {len(best_rows)} rows to results/best_per_experiment.csv")


if __name__ == "__main__":
    main()
