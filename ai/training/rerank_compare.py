"""Re-rank existing generation samples by Sethares roughness; emit comparison CSV.

Reads ``ai/results/generation_comparison.csv`` (from generate_compare.py) and,
within each (prompt, experiment, epoch) group, ranks the 3 samples by mean
Sethares roughness ascending. Emits ``ai/results/rerank_comparison.csv``
with original sample number, rerank position, score, and a compact
chord-only summary so a human can A/B the current default (sample 1) against
the re-rank winner.

Run from ``ai/`` with::

    uv run python -m training.rerank_compare
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

_AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AI_ROOT))

from physics.rerank import sequence_roughness  # noqa: E402

DEFAULT_INPUT = _AI_ROOT / "results" / "generation_comparison.csv"
DEFAULT_OUTPUT = _AI_ROOT / "results" / "rerank_comparison.csv"

_SUMMARY_CHORDS = 16  # cap chord-summary tokens for readability


def _chord_summary(tokens: list[str], cap: int = _SUMMARY_CHORDS) -> str:
    chords = [t for t in tokens if t and not t.startswith(("[", "<"))]
    snippet = chords[:cap]
    suffix = " …" if len(chords) > cap else ""
    return " ".join(snippet) + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--root-octave", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with args.input.open() as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["prompt"], r["experiment"], r["epoch"])].append(r)

    out_rows = []
    for (prompt, experiment, epoch), samples in sorted(groups.items()):
        scored = []
        for s in samples:
            tokens = ast.literal_eval(s["generated_text"])
            total, n = sequence_roughness(tokens, args.root_octave)
            mean = total / n if n > 0 else float("nan")
            scored.append({
                "sample": int(s["sample"]),
                "tokens": tokens,
                "total_roughness": total,
                "n_scored": n,
                "mean_roughness": mean,
            })
        scored.sort(key=lambda x: (x["mean_roughness"], x["sample"]))
        for new_rank, s in enumerate(scored, start=1):
            out_rows.append({
                "prompt": prompt,
                "experiment": experiment,
                "epoch": epoch,
                "original_sample": s["sample"],
                "rerank_position": new_rank,
                "mean_roughness": f"{s['mean_roughness']:.4f}",
                "total_roughness": f"{s['total_roughness']:.4f}",
                "n_scored_chords": s["n_scored"],
                "chord_summary": _chord_summary(s["tokens"]),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
