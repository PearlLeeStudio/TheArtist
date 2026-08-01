#!/usr/bin/env bash
# Sequential post-multi-seed pipeline.
#
# Runs after run_multi_seed_sweep.sh finishes:
#   1. aggregate_multi_seed.py        (eval all 33 chosen-rank cells, build CSV+MD)
#   2. decoding_artifacts.py          (sample F1 vs LoRA per genre, count artifacts)
#   3. run_peft_compare_sweep.sh      (train BitFit / IA3 / country full-FT, 7 runs)
#   4. eval_peft_compare.py           (test split, all methods)
#   5. system_ablation.py             (stitch CSVs into ablation summary)
#
# RESUMABLE: each step skips if its expected output CSV is already present.
# Delete the corresponding file in ai/results/ to force re-run. The PEFT
# training sweep (step 3) is resumable internally via skip_if_done.
set -e
cd "$(dirname "$0")/.."
echo "POST-MULTI-SEED pipeline starting:" $(date)

skip_if_csv() {
  local label="$1"; local path="$2"
  if [[ -s "$path" ]]; then
    echo "    SKIP $label — output exists at $path (delete to force re-run)"
    return 0
  fi
  return 1
}

echo
echo '=== 1. Aggregate multi-seed runs ==='
if ! skip_if_csv "step 1" "results/lora_multi_seed.csv"; then
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/aggregate_multi_seed.py
fi

echo
echo '=== 2. Decoding artifact quantification ==='
if ! skip_if_csv "step 2" "results/decoding_artifacts.csv"; then
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/decoding_artifacts.py
fi

echo
echo '=== 3. PEFT comparison sweep (7 runs, resumable per cell) ==='
bash training/run_peft_compare_sweep.sh

echo
echo '=== 4. PEFT comparison evaluation (test split, all methods) ==='
if ! skip_if_csv "step 4" "results/peft_compare.csv"; then
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/eval_peft_compare.py
fi

echo
echo '=== 5. Auto system ablation summary ==='
if ! skip_if_csv "step 5" "results/system_ablation.csv"; then
  UV_CACHE_DIR=/tmp/uv-cache uv run python analysis/system_ablation.py
fi

echo
echo "POST-MULTI-SEED pipeline finished:" $(date)
