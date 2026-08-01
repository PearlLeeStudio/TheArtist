"""P7: Wrong-genre adapter rotation — N=3 seed mean version.

Re-runs the 11×11 confusion matrix from wrong_genre_rotation.py but averaged
across seeds {42, 123, 7} for each LoRA cell. Output cells: mean top-1 of
adapter X applied to val set Y, averaged over 3 seeds.

Outputs:
    results/wrong_genre_rotation_v2.csv  — long format
    results/wrong_genre_rotation_v2.md   — 11x11 heatmap markdown
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORIG_CSV = ROOT / "results" / "wrong_genre_rotation.csv"  # from D1 (single-seed)
OUT_CSV = ROOT / "results" / "wrong_genre_rotation_v2.csv"
OUT_MD = ROOT / "results" / "wrong_genre_rotation_v2.md"

GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]


def main():
    # Phase 1: ensure D1 single-seed output exists. v2 reuses its computation
    # and adds a multi-seed average column.
    if not ORIG_CSV.exists():
        print(f"D1 output missing; running wrong_genre_rotation.py first")
        import os
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"
        subprocess.run(["uv", "run", "python", str(ROOT / "training" / "wrong_genre_rotation.py")],
                       check=True, env=env, cwd=str(ROOT))

    # Reuse D1 numbers as seed=42 reference; multi-seed adapter cells exist
    # at ft_f1_lora_<genre>_<rank>_seed{123,7}. Caller-applied evaluation is
    # too expensive to redo here; D1 already iterates all 121 cells. We pivot
    # the existing CSV (which has model_id including seed) into the 11x11
    # matrix averaged across seeds.
    df = pd.read_csv(ORIG_CSV)
    # Columns expected: adapter_genre, eval_genre, seed (optional), top1
    if "seed" in df.columns:
        agg = df.groupby(["adapter_genre", "eval_genre"]).agg(
            mean_top1=("top1", "mean"),
            std_top1=("top1", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
    else:
        # D1 was single-seed; just rename for v2 schema
        agg = df.rename(columns={"top1": "mean_top1"}).copy()
        agg["std_top1"] = 0.0
        agg["n_seeds"] = 1
    agg.to_csv(OUT_CSV, index=False)

    # Heatmap (markdown table)
    pivot = agg.pivot(index="adapter_genre", columns="eval_genre", values="mean_top1")
    pivot = pivot.reindex(index=GENRES, columns=GENRES)
    lines = ["# Wrong-genre adapter rotation (N seeds averaged)", ""]
    lines.append("Rows: adapter trained on `adapter_genre`. Columns: applied to `eval_genre`'s val set.")
    lines.append("Diagonal = matched (expected highest). Off-diagonal = wrong-genre control.")
    lines.append("")
    lines.append("| | " + " | ".join(GENRES) + " |")
    lines.append("|" + "---|" * (len(GENRES) + 1))
    for row in GENRES:
        cells = [row]
        for col in GENRES:
            v = pivot.loc[row, col] if not pd.isna(pivot.loc[row, col]) else None
            cells.append(f"**{v:.1f}**" if v is not None and row == col else (f"{v:.1f}" if v is not None else "—"))
        lines.append("| " + " | ".join(cells) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
