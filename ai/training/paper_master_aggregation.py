"""P15: Final aggregation table — paper §6 master.

Combines all phase 2 + phase 3 outputs into a single master table used as
the §6 results headline. Sources:
  - peft_compare_all.csv (165-cell grid)
  - method_pair_significance_corrected.csv (Wilcoxon + Holm/BH)
  - bootstrap_ci_song_level.csv
  - macro_summary.csv (per-method per-genre mean ± std)
  - wrong_genre_rotation_v2.csv (11×11 diagonal-dominance check)
  - generated_output_stats_v2.csv
  - real_song_paired.csv
  - dataset_validity.csv
  - per_genre_stats.csv
  - control_token / phase0_ablation / stage_a / stage_b cells (new eval CSVs
    that should be present after E1-F2 training; aggregated by eval_peft_compare
    style script if it exists)

Outputs:
    results/paper_master_table.csv  — concatenated long-format
    results/paper_master_table.md   — paper-ready summary with section headers
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "paper_master_table.csv"
OUT_MD = RESULTS / "paper_master_table.md"

SOURCES = [
    ("Canonical 165-cell PEFT grid", "peft_compare_all.csv"),
    ("Macro summary", "macro_summary.csv"),
    ("Method significance (Holm/BH)", "method_pair_significance_corrected.csv"),
    ("Song-level bootstrap CI", "bootstrap_ci_song_level.csv"),
    ("Wrong-genre rotation v2", "wrong_genre_rotation_v2.csv"),
    ("Generated-output stats v2", "generated_output_stats_v2.csv"),
    ("Real-song paired analysis", "real_song_paired.csv"),
    ("Dataset validity", "dataset_validity.csv"),
    ("Per-genre stats", "per_genre_stats.csv"),
    ("E1 control-token baseline", "control_token_baseline.csv"),
    ("E2 Phase 0 + Full FT", "phase0_full_ft.csv"),
    ("E3 Phase 0 + LoRA", "phase0_lora.csv"),
    ("F1 Stage A controlled funk-size", "controlled_funk_size.csv"),
    ("F2 Stage B epoch sensitivity", "epoch_sensitivity.csv"),
]


def main():
    md_lines = ["# Paper master table — phase 2 + phase 3 aggregation", ""]
    long_rows = []
    for label, fname in SOURCES:
        path = RESULTS / fname
        md_lines.append(f"## {label}")
        md_lines.append("")
        if not path.exists():
            md_lines.append(f"_(MISSING: {fname})_")
            md_lines.append("")
            continue
        try:
            df = pd.read_csv(path)
            md_lines.append(df.head(50).to_markdown(index=False))
            md_lines.append("")
            df["_source"] = label
            long_rows.append(df.assign(_source=label))
        except Exception as e:
            md_lines.append(f"_(ERROR reading {fname}: {e})_")
            md_lines.append("")

    if long_rows:
        # Each source has different columns; just concat with NaN fill
        combined = pd.concat(long_rows, ignore_index=True, sort=False)
        combined.to_csv(OUT_CSV, index=False)

    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_MD}")
    if long_rows:
        print(f"Wrote {OUT_CSV} ({len(combined)} total rows from {len(long_rows)} sources)")


if __name__ == "__main__":
    main()
