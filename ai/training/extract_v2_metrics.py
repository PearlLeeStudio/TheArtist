"""Quick log → CSV extractor for v2 chain cells.

Scans ai/logs/ for per-cell training logs and pulls the last epoch's
train_loss / val_loss / top1 / top5. No model loading, no GPU touch —
purely greps the existing log files, so safe to run while the chain
is in flight.

Usage (from ai/):
    uv run python training/extract_v2_metrics.py
    uv run python training/extract_v2_metrics.py --output results/v2_a_partial.csv
    uv run python training/extract_v2_metrics.py --filter bitfit
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+train=([\d.]+)\s+val=([\d.]+)\s+top1=([\d.]+)%\s+top5=([\d.]+)%"
)


def parse_run_name(log_path: Path) -> str:
    # logs/ft_f1_bitfit_blues_20260520_172613.log → ft_f1_bitfit_blues
    stem = log_path.stem
    parts = stem.split("_")
    while parts and (parts[-1].isdigit() or len(parts[-1]) == 8):
        parts.pop()
    return "_".join(parts)


def parse_method_genre_seed(run_name: str) -> tuple[str, str, int]:
    # ft_f1_<method>_<genre>[_seed<S>] OR ft_f1_lora_<genre>[_r<R>][_seed<S>]
    parts = run_name.split("_")
    if parts[0] == "ft" and parts[1] == "f1":
        parts = parts[2:]
    seed = 42
    if parts and parts[-1].startswith("seed"):
        try:
            seed = int(parts[-1].removeprefix("seed"))
        except ValueError:
            pass
        parts = parts[:-1]
    # strip trailing r<N> from LoRA
    if parts and re.fullmatch(r"r\d+", parts[-1]):
        parts = parts[:-1]
    if not parts:
        return ("?", "?", seed)
    if parts[0] == "lora":
        method = "lora"
        genre = "_".join(parts[1:])
    elif parts[0] == "full":
        method = "full_ft" if len(parts) > 1 and parts[1] == "ft" else "full"
        genre = "_".join(parts[2:]) if method == "full_ft" else "_".join(parts[1:])
    elif parts[0] in {"bitfit", "ia3", "prefix"}:
        method = parts[0]
        genre = "_".join(parts[1:])
    else:
        method = parts[0]
        genre = "_".join(parts[1:])
    return (method, genre, seed)


def extract_last_epoch(log_path: Path) -> dict | None:
    text = log_path.read_text(errors="ignore")
    matches = EPOCH_RE.findall(text)
    if not matches:
        return None
    epoch, train, val, top1, top5 = matches[-1]
    return {
        "last_epoch": int(epoch),
        "train_loss": float(train),
        "val_loss": float(val),
        "top1": float(top1),
        "top5": float(top5),
        "n_epochs_logged": len(matches),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default=None,
                    help="Optional path to write CSV. If omitted, prints to stdout.")
    ap.add_argument("--filter", "-f", default=None,
                    help="Substring filter on run name (e.g., 'bitfit' or 'rock').")
    args = ap.parse_args()

    if not LOG_DIR.exists():
        print(f"Log dir not found: {LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    # Group log files by run_name, pick most recent per run.
    latest_per_run: dict[str, Path] = {}
    for log_path in LOG_DIR.glob("ft_f1_*.log"):
        rn = parse_run_name(log_path)
        if args.filter and args.filter not in rn:
            continue
        if rn not in latest_per_run or log_path.stat().st_mtime > latest_per_run[rn].stat().st_mtime:
            latest_per_run[rn] = log_path

    rows = []
    for rn, log_path in sorted(latest_per_run.items()):
        method, genre, seed = parse_method_genre_seed(rn)
        metrics = extract_last_epoch(log_path)
        if metrics is None:
            continue
        rows.append({
            "run_name": rn,
            "method": method,
            "genre": genre,
            "seed": seed,
            **metrics,
            "log_file": log_path.name,
        })

    if not rows:
        print("No matching cells with epoch metrics found.", file=sys.stderr)
        sys.exit(0)

    fieldnames = ["run_name", "method", "genre", "seed",
                  "last_epoch", "n_epochs_logged",
                  "train_loss", "val_loss", "top1", "top5", "log_file"]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
