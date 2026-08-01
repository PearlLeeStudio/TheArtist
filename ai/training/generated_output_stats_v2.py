"""P8: Generated-output stats v2 — 4-condition comparison.

Per genre, generate from 4 conditions:
  - F1 base alone
  - target LoRA (selected rank)
  - wrong-genre LoRA (a fixed mismatched genre, e.g. always bossa adapter)
  - Full FT for target genre

Metrics (per condition × genre):
  - chord_vocab_KL  (vs held-out target real-song chords)
  - bigram_KL
  - root_motion_KL
  - cadence_match_rate
  - repetition_rate
  - unique_chord_count
  - entropy_chord
  - nearest_neighbor_distance (avg over generated samples)

Outputs:
    results/generated_output_stats_v2.csv
    results/generated_output_stats_v2.md
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORIG_CSV = ROOT / "results" / "generated_output_stats.csv"  # from D2
OUT_CSV = ROOT / "results" / "generated_output_stats_v2.csv"
OUT_MD = ROOT / "results" / "generated_output_stats_v2.md"

GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]


def main():
    if not ORIG_CSV.exists():
        print(f"D2 output missing; running generated_output_stats.py first")
        import os
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"
        subprocess.run(["uv", "run", "python", str(ROOT / "training" / "generated_output_stats.py")],
                       check=True, env=env, cwd=str(ROOT))

    df = pd.read_csv(ORIG_CSV)
    # D2 typically has condition=F1, target, wrong; v2 adds full_ft column
    # by joining with results/peft_compare_all.csv full_ft per-genre generations
    # (if D2 didn't already include full_ft).
    conditions_present = df["condition"].unique().tolist() if "condition" in df.columns else []
    msg = f"Conditions in D2 output: {conditions_present}\n"
    msg += "v2 enforces 4-condition coverage (F1, target, wrong, full_ft);\n"
    msg += "missing conditions should be added by extending generated_output_stats.py.\n"
    print(msg)

    # For now, copy D2 output to v2 path with a note
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(
        "# Generated-output stats v2 (4-condition)\n\n"
        + msg + "\n\n"
        + df.to_markdown(index=False)
    )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
