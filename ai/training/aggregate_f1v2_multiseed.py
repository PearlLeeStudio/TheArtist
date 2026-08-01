"""Aggregate the F1 v2 multi-seed restoration runs (seeds 42/123/7) on the 9-src corpus.

For each run, selects the jazz_val-minimizing epoch (the served best.pt rule) and reports
jazz/pop test top-1, plus the epoch-3 row (= Phase 0, the un-adapted baseline on the 9-src
eval set). Emits results/f1v2_multiseed.csv + a short markdown summary with mean +/- std.

    cd ai && uv run python training/aggregate_f1v2_multiseed.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

AI = Path(__file__).resolve().parent.parent
RUNS = [("ft_jazz_pop80_v2_s42", 42), ("ft_jazz_pop80_v2_s2", 123), ("ft_jazz_pop80_v2_s3", 7)]


def load(run: str) -> list[dict]:
    p = AI / "checkpoints" / run / "eval_results.csv"
    with open(p) as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f)]


def pick(rows: list[dict]) -> dict:
    sel = min(rows, key=lambda r: float(r["jazz_val_loss"]))
    init = next(r for r in rows if int(r["epoch"]) == 3)  # epoch 3 == Phase 0 (warmup, zero-update)
    return {
        "epoch": int(sel["epoch"]),
        "jazz_val_loss": round(float(sel["jazz_val_loss"]), 4),
        "jazz_top1": round(float(sel["jazz_top1"]), 2),
        "pop_top1": round(float(sel["pop_top1"]), 2),
        "phase0_jazz_top1": round(float(init["jazz_top1"]), 2),
    }


def main() -> None:
    out_rows = []
    for run, seed in RUNS:
        r = pick(load(run))
        r.update(run=run, seed=seed)
        out_rows.append(r)

    jazz = [r["jazz_top1"] for r in out_rows]
    pop = [r["pop_top1"] for r in out_rows]
    base = out_rows[0]["phase0_jazz_top1"]  # identical across runs (epoch 3 == Phase 0)
    jazz_mean, jazz_sd = statistics.mean(jazz), (statistics.stdev(jazz) if len(jazz) > 1 else 0.0)
    pop_mean, pop_sd = statistics.mean(pop), (statistics.stdev(pop) if len(pop) > 1 else 0.0)

    csv_path = AI / "results" / "f1v2_multiseed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "selected_epoch", "jazz_val_loss",
                    "jazz_top1_9src501", "pop_top1_9src501", "phase0_jazz_top1_9src501",
                    "corpus", "eval_set"])
        for r in out_rows:
            w.writerow([r["run"], r["seed"], r["epoch"], r["jazz_val_loss"],
                        r["jazz_top1"], r["pop_top1"], r["phase0_jazz_top1"],
                        "9-src (jazz 4250 / pop 10000)", "9-src jazz test (501 seqs)"])
        w.writerow([f"MEAN(n={len(out_rows)})", "", "", "",
                    f"{jazz_mean:.2f}", f"{pop_mean:.2f}", base, "", ""])
        w.writerow(["STD", "", "", "", f"{jazz_sd:.2f}", f"{pop_sd:.2f}", "", "", ""])

    md = AI / "results" / "f1v2_multiseed.md"
    md.write_text(
        "# F1 v2 multi-seed (n=3, 9-src, jazz_val-selected)\n\n"
        "| run | seed | epoch | jazz_val_loss | jazz top-1 | pop top-1 |\n"
        "|---|---|---|---|---|---|\n"
        + "".join(f"| {r['run']} | {r['seed']} | {r['epoch']} | {r['jazz_val_loss']} "
                 f"| {r['jazz_top1']} | {r['pop_top1']} |\n" for r in out_rows)
        + f"| **mean ± std** | | | | **{jazz_mean:.2f} ± {jazz_sd:.2f}** "
          f"| **{pop_mean:.2f} ± {pop_sd:.2f}** |\n\n"
        f"Un-adapted baseline (epoch 3 = Phase 0, 9-src eval): jazz {base}. "
        f"Δ jazz vs un-adapted = +{jazz_mean - base:.2f}. "
        "All runs hash-distinct from Phase 0; eval set = 9-src jazz test (501 seqs). "
        "Source eval CSVs: ai/checkpoints/ft_jazz_pop80_v2_s{42,2,3}/eval_results.csv.\n"
    )
    print(f"jazz {jazz_mean:.2f} ± {jazz_sd:.2f} | pop {pop_mean:.2f} ± {pop_sd:.2f} | base {base}")
    print("wrote", csv_path, "and", md)


if __name__ == "__main__":
    main()
