"""Selection-criterion ablation over the F-series fine-tune runs (novelty memo #1).

Question: how often does the best-checkpoint selection criterion change the
selected epoch, and what does each choice cost in jazz/pop metrics? This is
the n=1 defense for the "selection silently discards adaptation" finding:
the F1 artifact is the headline case; this table shows the behavior of the
same recipe across the whole rehearsal-mix sweep.

Criteria compared per run, over its per-epoch eval_results.csv:
  mixed_val   — min val_loss on the run's own validation mix (the v1 rule)
  jazz_loss   — min per-epoch jazz loss. For the v1-era runs only the jazz
                TEST loss was logged per epoch, so for them this criterion is
                RETROSPECTIVE (not a deployable rule — test leakage); for
                ft_jazz_pop80_v2 the leakage-free jazz_val_loss column is used.

Coverage: the 6 fine-tune runs (F1–F5 + F1 v2). Phase 0 is excluded (no
fine-tune, nothing to select against). The 165-cell PEFT grid is excluded —
its per-epoch training logs lived on the lost GPU laptop; only last-epoch
summaries survive (results/v2_all_current.csv).

Output: results/selection_criterion_ablation_20260612.csv + .md

Usage (from ai/): uv run python -m analysis.selection_criterion_ablation
"""
from __future__ import annotations

import csv
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent.parent
RESULTS = AI_ROOT / "results"
CKPT = AI_ROOT / "checkpoints"
STAMP = "20260612"

RUNS = [
    # (label, checkpoint dir, jazz-criterion column, leakage-free?)
    ("F1 (ft_jazz_pop80, v1)", "ft_jazz_pop80", "jazz_loss", False),
    ("F1 v2 (ft_jazz_pop80_v2)", "ft_jazz_pop80_v2", "jazz_val_loss", True),
    ("F2 (ft_jazz_pop67)", "ft_jazz_pop67", "jazz_loss", False),
    ("F3 (ft_jazz_pop50)", "ft_jazz_pop50", "jazz_loss", False),
    ("F4 (ft_jazz_pop29)", "ft_jazz_pop29", "jazz_loss", False),
    ("F5 (ft_jazz_only)", "ft_jazz_only", "jazz_loss", False),
]


def main() -> None:
    out_rows = []
    for label, sub, jazz_col, clean in RUNS:
        path = CKPT / sub / "eval_results.csv"
        if not path.exists():
            print(f"  SKIP {label}: {path} missing")
            continue
        epochs = []
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    epochs.append({
                        "epoch": int(r["epoch"]),
                        "val_loss": float(r["val_loss"]),
                        "jazz_crit": float(r[jazz_col]),
                        "jazz_top1": float(r["jazz_top1"]),
                        "pop_top1": float(r["pop_top1"]),
                    })
                except (KeyError, ValueError):
                    continue
        if not epochs:
            print(f"  SKIP {label}: no parsable epochs")
            continue

        by_mixed = min(epochs, key=lambda e: e["val_loss"])
        by_jazz = min(epochs, key=lambda e: e["jazz_crit"])
        out_rows.append({
            "run": label,
            "n_epochs": len(epochs),
            "epoch_mixed_val": by_mixed["epoch"],
            "epoch_jazz_crit": by_jazz["epoch"],
            "criterion_changes_pick": by_mixed["epoch"] != by_jazz["epoch"],
            "jazz_criterion": jazz_col,
            "leakage_free_criterion": clean,
            "jazz_top1_at_mixed": by_mixed["jazz_top1"],
            "jazz_top1_at_jazz": by_jazz["jazz_top1"],
            "delta_jazz_top1": round(by_jazz["jazz_top1"] - by_mixed["jazz_top1"], 2),
            "pop_top1_at_mixed": by_mixed["pop_top1"],
            "pop_top1_at_jazz": by_jazz["pop_top1"],
            "delta_pop_top1": round(by_jazz["pop_top1"] - by_mixed["pop_top1"], 2),
        })
        flag = "CHANGES" if by_mixed["epoch"] != by_jazz["epoch"] else "same"
        print(f"  {label:28s} mixed→ep{by_mixed['epoch']} jazz→ep{by_jazz['epoch']} "
              f"[{flag}] Δjazz_top1={out_rows[-1]['delta_jazz_top1']:+.2f} "
              f"Δpop_top1={out_rows[-1]['delta_pop_top1']:+.2f}")

    out_csv = RESULTS / f"selection_criterion_ablation_{STAMP}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    n_change = sum(1 for r in out_rows if r["criterion_changes_pick"])
    md = [
        f"# Selection-criterion ablation — F-series fine-tunes ({STAMP})\n",
        "Per run: epoch picked by min mixed-val loss (the v1 rule) vs by min",
        "jazz loss. v1-era runs log only per-epoch jazz TEST loss, so their",
        "jazz criterion is **retrospective** (test leakage if deployed); the",
        "v2 run uses the leakage-free `jazz_val_loss`. The 165-cell PEFT grid",
        "is out of coverage (per-epoch logs lost with the GPU laptop; only",
        "last-epoch summaries survive in `v2_all_current.csv`).\n",
        f"**Headline: the criterion changes the selected checkpoint in "
        f"{n_change}/{len(out_rows)} fine-tune runs.**\n",
        "| run | mixed-val pick | jazz pick | changes? | Δ jazz top-1 (pp) | Δ pop top-1 (pp) | jazz criterion |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for r in out_rows:
        md.append(
            f"| {r['run']} | ep{r['epoch_mixed_val']} | ep{r['epoch_jazz_crit']} | "
            f"{'**yes**' if r['criterion_changes_pick'] else 'no'} | "
            f"{r['delta_jazz_top1']:+.2f} | {r['delta_pop_top1']:+.2f} | "
            f"{r['jazz_criterion']}{'' if r['leakage_free_criterion'] else ' (retrospective)'} |")
    md.append("\nSource: per-run `ai/checkpoints/<run>/eval_results.csv`. "
              "Script: `analysis/selection_criterion_ablation.py`.\n")
    (RESULTS / f"selection_criterion_ablation_{STAMP}.md").write_text("\n".join(md))
    print(f"\nWrote selection_criterion_ablation_{STAMP}.csv/.md")


if __name__ == "__main__":
    main()
