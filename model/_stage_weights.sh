#!/bin/bash
# Stage trained weights and inference code into each model/{run}/ folder
# so the folder is ready to upload to HuggingFace.
#
# Reads from: ai/checkpoints/{run}/, ai/training/
# Writes to:  model/{run}/
#
# Usage:  bash model/_stage_weights.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -a runs=(
  "phase0_pop_baseline"
  "ft_jazz_pop80"
  "ft_jazz_pop80_v2"
  "ft_jazz_pop67"
  "ft_jazz_pop50"
  "ft_jazz_pop29"
  "ft_jazz_only"
)

declare -a lora_genres=(
  "country" "funk" "gospel" "rnb_soul" "hip_hop"
  "electronic" "folk" "classical" "rock" "blues" "bossa"
)

for run in "${runs[@]}"; do
  src="$REPO_DIR/ai/checkpoints/$run"
  dst="$SCRIPT_DIR/$run"

  if [ ! -d "$src" ]; then
    echo "Skipping $run: checkpoint directory not found at $src"
    continue
  fi
  if [ ! -d "$dst" ]; then
    echo "Skipping $run: model card folder not found at $dst"
    continue
  fi

  # Hub-downloaded checkpoint dirs (model/_download.py) carry only
  # best.pt/config.json/eval_results.csv — skip those instead of dying
  # under set -e; staging needs a full training dir (with tokenizer.json).
  missing=""
  for req in best.pt config.json tokenizer.json; do
    [ -f "$src/$req" ] || missing="$missing $req"
  done
  if [ -n "$missing" ]; then
    echo "Skipping $run: missing$missing (hub-download dir? stage from a full training dir)"
    continue
  fi

  echo "Staging $run ..."
  cp "$src/best.pt"          "$dst/best.pt"
  cp "$src/config.json"      "$dst/config.json"
  cp "$src/tokenizer.json"   "$dst/tokenizer.json"
  if [ -f "$src/eval_results.csv" ]; then
    cp "$src/eval_results.csv" "$dst/eval_results.csv"
  fi
  cp "$REPO_DIR/ai/training/model.py"     "$dst/model.py"
  cp "$REPO_DIR/ai/training/tokenizer.py" "$dst/tokenizer.py"
done

# Stage LoRA adapters. Layout per HF/peft convention: adapter_*.{json,safetensors}
# at repo root, plus the tokenizer extension and base-model code so a single
# from_pretrained call works without separately fetching the F1 base.
for genre in "${lora_genres[@]}"; do
  run="ft_f1_lora_$genre"
  src="$REPO_DIR/ai/checkpoints/$run"
  dst="$SCRIPT_DIR/$run"

  if [ ! -d "$src" ]; then
    echo "Skipping $run: checkpoint directory not found at $src"
    continue
  fi

  missing=""
  for req in adapter/adapter_config.json adapter/adapter_model.safetensors \
             config.json tokenizer.json embedding_extension.pt; do
    [ -f "$src/$req" ] || missing="$missing $req"
  done
  if [ -n "$missing" ]; then
    echo "Skipping $run: missing$missing (hub-download dir? stage from a full training dir)"
    continue
  fi
  mkdir -p "$dst"

  echo "Staging $run ..."
  cp "$src/adapter/adapter_config.json"        "$dst/adapter_config.json"
  cp "$src/adapter/adapter_model.safetensors"  "$dst/adapter_model.safetensors"
  cp "$src/config.json"                        "$dst/config.json"
  cp "$src/tokenizer.json"                     "$dst/tokenizer.json"
  cp "$src/embedding_extension.pt"             "$dst/embedding_extension.pt"
  cp "$REPO_DIR/ai/training/model.py"          "$dst/model.py"
  cp "$REPO_DIR/ai/training/tokenizer.py"      "$dst/tokenizer.py"
done

echo
echo "All checkpoints staged. Folder contents:"
for run in "${runs[@]}"; do
  dst="$SCRIPT_DIR/$run"
  if [ -f "$dst/best.pt" ]; then
    echo
    echo "  $run/"
    ls -lh "$dst/" | awk 'NR>1 {printf "    %s  %s\n", $5, $9}'
  fi
done
for genre in "${lora_genres[@]}"; do
  run="ft_f1_lora_$genre"
  dst="$SCRIPT_DIR/$run"
  if [ -f "$dst/adapter_model.safetensors" ]; then
    echo
    echo "  $run/"
    ls -lh "$dst/" | awk 'NR>1 {printf "    %s  %s\n", $5, $9}'
  fi
done

echo
echo "Next: hf auth login  (write token from https://huggingface.co/settings/tokens)"
echo "Then: python model/_upload.py"
