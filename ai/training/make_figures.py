"""Generate paper figures from aggregated CSVs.

Outputs to paper_assets/:
  - figure1_mix_vs_accuracy.png — Pop mix size vs per-genre top-1
  - figure2_learning_curves.png — per-epoch pop/jazz accuracy per run

Usage:
    cd ai
    uv run python training/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_DIR / "ai" / "results" / "all_experiments.csv"
OUT_DIR = PROJECT_DIR / "paper_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order & pop-mix labels
RUNS_ORDER = [
    ("phase0_pop_baseline", None, "Phase 0\nbaseline"),
    ("ft_jazz_only",        0,    "F5\n(0)"),
    ("ft_jazz_pop29",       1000, "F4\n(1K)"),
    ("ft_jazz_pop50",       2500, "F3\n(2.5K)"),
    ("ft_jazz_pop67",       5000, "F2\n(5K)"),
    ("ft_jazz_pop80",       10000,"F1\n(10K)"),
]

# Best (used for Figure 1): selected in the paper Section 5.1
BEST_EPOCH = {
    "phase0_pop_baseline": 2,
    "ft_jazz_pop80":       6,
    "ft_jazz_pop67":       4,
    "ft_jazz_pop50":       9,
    "ft_jazz_pop29":       6,
    "ft_jazz_only":        7,
}


def load_rows() -> list[dict]:
    with open(RESULTS) as f:
        return list(csv.DictReader(f))


def f(x: str) -> float:
    return float(x)


# ---------- Figure 1: pop mix vs accuracy ----------

def figure1(rows: list[dict]) -> None:
    # Pick best-epoch rows per experiment
    best_rows = {}
    for r in rows:
        exp = r["experiment"]
        if int(r["epoch"]) == BEST_EPOCH.get(exp):
            best_rows[exp] = r

    # Build x/y arrays in pop-mix order
    mix_order = [
        ("ft_jazz_only",   0),
        ("ft_jazz_pop29",  1000),
        ("ft_jazz_pop50",  2500),
        ("ft_jazz_pop67",  5000),
        ("ft_jazz_pop80",  10000),
    ]
    x = [m for _, m in mix_order]
    pop = [f(best_rows[e]["pop_top1"]) for e, _ in mix_order]
    jazz = [f(best_rows[e]["jazz_top1"]) for e, _ in mix_order]

    baseline_pop = f(best_rows["phase0_pop_baseline"]["pop_top1"])
    baseline_jazz = f(best_rows["phase0_pop_baseline"]["jazz_top1"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(x, pop, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Pop test top-1")
    ax.plot(x, jazz, "s-", color="#d62728", linewidth=2, markersize=8, label="Jazz test top-1")

    # Baseline dashed lines
    ax.axhline(baseline_pop, linestyle="--", color="#1f77b4", alpha=0.4, linewidth=1)
    ax.axhline(baseline_jazz, linestyle="--", color="#d62728", alpha=0.4, linewidth=1)
    ax.text(10200, baseline_pop + 0.3, f"pop baseline ({baseline_pop:.2f}%)",
            color="#1f77b4", fontsize=8, alpha=0.8)
    ax.text(10200, baseline_jazz - 1.2, f"jazz baseline ({baseline_jazz:.2f}%)",
            color="#d62728", fontsize=8, alpha=0.8)

    # Highlight sweet spot (F3 at 2500)
    ax.axvspan(2000, 3000, alpha=0.08, color="green")
    ax.text(2500, 75, "sweet spot\n(F3)", ha="center", fontsize=9, color="green", weight="bold")

    ax.set_xlabel("Pop rehearsal mix size (number of songs)")
    ax.set_ylabel("Top-1 chord accuracy (%)")
    ax.set_title("Figure 1. Per-genre test accuracy vs. pop rehearsal mix size")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m//1000}K" if m >= 1000 else str(m) for m in x])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")

    ax.set_ylim(70, 88)

    # Annotate points
    for xi, yi_pop, yi_jazz in zip(x, pop, jazz):
        ax.annotate(f"{yi_pop:.2f}", xy=(xi, yi_pop), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color="#1f77b4")
        ax.annotate(f"{yi_jazz:.2f}", xy=(xi, yi_jazz), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8, color="#d62728")

    plt.tight_layout()
    out = OUT_DIR / "figure1_mix_vs_accuracy.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------- Figure 2: learning curves per run ----------

def figure2(rows: list[dict]) -> None:
    # Group rows by experiment
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], []).append(r)
    for k in by_exp:
        by_exp[k].sort(key=lambda r: int(r["epoch"]))

    fig, (ax_pop, ax_jazz) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    colors = {
        "phase0_pop_baseline": "#7f7f7f",
        "ft_jazz_pop80":       "#1f77b4",
        "ft_jazz_pop67":       "#2ca02c",
        "ft_jazz_pop50":       "#ff7f0e",
        "ft_jazz_pop29":       "#9467bd",
        "ft_jazz_only":        "#d62728",
    }
    labels = {
        "phase0_pop_baseline": "Phase 0 (pop baseline)",
        "ft_jazz_pop80":       "F1 (10K pop mix)",
        "ft_jazz_pop67":       "F2 (5K pop mix)",
        "ft_jazz_pop50":       "F3 (2.5K pop mix)",
        "ft_jazz_pop29":       "F4 (1K pop mix)",
        "ft_jazz_only":        "F5 (jazz only)",
    }

    for exp_key in ["phase0_pop_baseline", "ft_jazz_pop80", "ft_jazz_pop67",
                    "ft_jazz_pop50", "ft_jazz_pop29", "ft_jazz_only"]:
        rr = by_exp.get(exp_key, [])
        if not rr:
            continue
        ep = [int(r["epoch"]) for r in rr]
        pop = [f(r["pop_top1"]) for r in rr]
        jazz = [f(r["jazz_top1"]) for r in rr]

        style = "--" if exp_key == "phase0_pop_baseline" else "-"
        ax_pop.plot(ep, pop, style, color=colors[exp_key], marker="o", markersize=5,
                    linewidth=2, label=labels[exp_key], alpha=0.9)
        ax_jazz.plot(ep, jazz, style, color=colors[exp_key], marker="o", markersize=5,
                     linewidth=2, label=labels[exp_key], alpha=0.9)

    # Baseline horizontal reference on both panels
    baseline_rows = by_exp["phase0_pop_baseline"]
    final_pop_baseline = f(baseline_rows[-1]["pop_top1"])
    final_jazz_baseline = f(baseline_rows[-1]["jazz_top1"])
    ax_pop.axhline(final_pop_baseline, color="#7f7f7f", linestyle=":", alpha=0.5)
    ax_jazz.axhline(final_jazz_baseline, color="#7f7f7f", linestyle=":", alpha=0.5)

    ax_pop.set_xlabel("Epoch")
    ax_pop.set_ylabel("Pop top-1 accuracy (%)")
    ax_pop.set_title("Pop test accuracy per epoch")
    ax_pop.grid(True, alpha=0.3)
    ax_pop.legend(fontsize=8, loc="lower left")

    ax_jazz.set_xlabel("Epoch")
    ax_jazz.set_ylabel("Jazz top-1 accuracy (%)")
    ax_jazz.set_title("Jazz test accuracy per epoch")
    ax_jazz.grid(True, alpha=0.3)
    ax_jazz.legend(fontsize=8, loc="lower right")

    fig.suptitle("Figure 2. Per-epoch learning curves (fine-tune starts at epoch 3)",
                 y=1.02, fontsize=11)

    plt.tight_layout()
    out = OUT_DIR / "figure2_learning_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------- Figure 3 (bonus): Pop-vs-Jazz tradeoff scatter ----------

def figure3(rows: list[dict]) -> None:
    best_rows = {}
    for r in rows:
        exp = r["experiment"]
        if int(r["epoch"]) == BEST_EPOCH.get(exp):
            best_rows[exp] = r

    fig, ax = plt.subplots(figsize=(6, 5.5))

    mapping = [
        ("phase0_pop_baseline", "#7f7f7f", "Phase 0\nbaseline"),
        ("ft_jazz_pop80",       "#1f77b4", "F1 (10K)"),
        ("ft_jazz_pop67",       "#2ca02c", "F2 (5K)"),
        ("ft_jazz_pop50",       "#ff7f0e", "F3 (2.5K)"),
        ("ft_jazz_pop29",       "#9467bd", "F4 (1K)"),
        ("ft_jazz_only",        "#d62728", "F5 (0)"),
    ]

    for exp_key, color, label in mapping:
        r = best_rows[exp_key]
        pop = f(r["pop_top1"])
        jazz = f(r["jazz_top1"])
        ax.scatter(pop, jazz, color=color, s=140, edgecolor="black", linewidth=1, zorder=3)
        ax.annotate(label, (pop, jazz), xytext=(8, 8), textcoords="offset points", fontsize=9)

    # Ideal region
    baseline_pop = f(best_rows["phase0_pop_baseline"]["pop_top1"])
    baseline_jazz = f(best_rows["phase0_pop_baseline"]["jazz_top1"])
    ax.axvline(baseline_pop, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(baseline_jazz, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(baseline_pop + 0.05, 73, "pop baseline", rotation=90, fontsize=8, color="gray")
    ax.text(82.3, baseline_jazz + 0.2, "jazz baseline", fontsize=8, color="gray")

    ax.set_xlabel("Pop top-1 accuracy (%)")
    ax.set_ylabel("Jazz top-1 accuracy (%)")
    ax.set_title("Figure 3. Pop vs. Jazz accuracy trade-off\n(upper-right = better on both)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "figure3_tradeoff_scatter.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    rows = load_rows()
    figure1(rows)
    figure2(rows)
    figure3(rows)
    print(f"All figures in {OUT_DIR}")
