#!/usr/bin/env bash
# Sequentially trains prefix-tuning runs for all 11 target genres.
# Extended from the original 3-genre version (Pearl plan 2026-05-18) so
# that paper §6.2 PEFT comparison has a complete prefix-tuning row.
# Skips a cell whose checkpoint already exists (resumable).
set -e
cd "$(dirname "$0")/.."
echo 'Prefix-tuning sweep starting:' $(date)

skip_if_done() {
  local rn="$1"
  if [[ -s "checkpoints/${rn}/prefix_state.pt" && -s "checkpoints/${rn}/embedding_extension.pt" ]]; then return 0; fi
  return 1
}

run_cell() {
  local idx="$1"; local rn="$2"; local cfg="$3"
  echo "[${idx}/11] ${cfg}"
  if skip_if_done "$rn"; then
    echo "    SKIP — checkpoint exists at checkpoints/${rn}"
  else
    UV_CACHE_DIR=/tmp/uv-cache uv run python training/lora_train.py --config "$cfg"
  fi
}

run_cell 1  ft_f1_prefix_classical  training/configs/lora/peft_compare/classical_prefix.yaml
run_cell 2  ft_f1_prefix_country    training/configs/lora/peft_compare/country_prefix.yaml
run_cell 3  ft_f1_prefix_rock       training/configs/lora/peft_compare/rock_prefix.yaml
run_cell 4  ft_f1_prefix_funk       training/configs/lora/peft_compare/funk_prefix.yaml
run_cell 5  ft_f1_prefix_gospel     training/configs/lora/peft_compare/gospel_prefix.yaml
run_cell 6  ft_f1_prefix_rnb_soul   training/configs/lora/peft_compare/rnb_soul_prefix.yaml
run_cell 7  ft_f1_prefix_hip_hop    training/configs/lora/peft_compare/hip_hop_prefix.yaml
run_cell 8  ft_f1_prefix_electronic training/configs/lora/peft_compare/electronic_prefix.yaml
run_cell 9  ft_f1_prefix_folk       training/configs/lora/peft_compare/folk_prefix.yaml
run_cell 10 ft_f1_prefix_blues      training/configs/lora/peft_compare/blues_prefix.yaml
run_cell 11 ft_f1_prefix_bossa      training/configs/lora/peft_compare/bossa_prefix.yaml

echo 'Prefix-tuning sweep finished:' $(date)
echo 'Now run: UV_CACHE_DIR=/tmp/uv-cache uv run python training/eval_peft_compare.py --all-genres'
