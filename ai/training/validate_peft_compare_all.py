"""P1: Validate that peft_compare_all.csv covers all 165 cells.

Required: 5 methods × 11 genres × 3 seeds = 165 rows.
Columns: method, genre, seed, epoch, trainable_params, val_loss, test_loss, top1, top5.

Outputs:
    results/peft_compare_all_validated.md  — coverage report + gap list
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "peft_compare_all.csv"
OUT = ROOT / "results" / "peft_compare_all_validated.md"

METHODS = ["bitfit", "ia3", "full", "prefix_tuning", "lora"]  # match eval_peft_compare.py names
GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
SEEDS = [42, 123, 7]


def main():
    if not CSV.exists():
        raise SystemExit(f"FAIL: {CSV} missing — run v2 chain step C first")

    seen = set()
    rows = []
    with CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            method = row.get("method", "").lower()
            # Skip F1 baseline rows (one per genre, no seed)
            if method == "f1 (no adapt)":
                continue
            seed_str = row.get("seed", "")
            try:
                seed = int(seed_str)
            except (ValueError, TypeError):
                continue
            key = (method, row.get("genre", "").lower(), seed)
            seen.add(key)

    expected = {(m, g, s) for m in METHODS for g in GENRES for s in SEEDS}
    missing = expected - seen
    extra = seen - expected

    lines = ["# peft_compare_all.csv validation report", ""]
    lines.append(f"- Required cells: 5 methods × 11 genres × 3 seeds = **165**")
    lines.append(f"- Found in CSV: **{len(seen)}** unique (method, genre, seed) triples")
    lines.append(f"- Missing: **{len(missing)}**")
    lines.append(f"- Extra/unexpected: **{len(extra)}**")
    lines.append("")

    if missing:
        lines.append("## Missing cells")
        for m, g, s in sorted(missing):
            lines.append(f"- {m} / {g} / seed={s}")
        lines.append("")
    if extra:
        lines.append("## Extra cells (not in canonical grid)")
        for m, g, s in sorted(extra):
            lines.append(f"- {m} / {g} / seed={s}")
        lines.append("")

    if not missing and not extra:
        lines.append("**PASS** — canonical 165-cell grid complete.")
    else:
        lines.append("**FAIL** — grid incomplete; re-run chain or check missing/extra entries.")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} cells missing from peft_compare_all.csv")
    print(f"PASS: 165/165 cells present")


if __name__ == "__main__":
    main()
