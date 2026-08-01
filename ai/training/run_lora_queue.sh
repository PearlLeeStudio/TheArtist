#!/bin/bash
# Sequential LoRA training queue.
# Pearl 2026-05-09 — kicks off after country LoRA completes.
# Order: smallest datasets first (~5min) for quick validation, then larger.
set -e

LOG=/tmp/lora_queue.log
COUNTRY_PID=${1:-5847}

cd "$(dirname "$0")/.."

echo "=== LoRA queue started at $(date) ===" | tee -a $LOG
echo "Waiting for country LoRA (PID $COUNTRY_PID) to finish..." | tee -a $LOG

# Poll until country training process exits (avoid GPU contention)
while kill -0 "$COUNTRY_PID" 2>/dev/null; do
  sleep 60
done
echo "Country LoRA done at $(date). Starting queue." | tee -a $LOG

# Order: small → large for quick validation + progressive long jobs.
# Skip classical (IMSLP data pending) and bossa (already partial in F1 vocab,
# can re-add after first wave validates).
GENRES=(funk gospel blues rnb_soul hip_hop bossa electronic folk rock)

for genre in "${GENRES[@]}"; do
  config="training/configs/lora/${genre}.yaml"
  if [ ! -f "$config" ]; then
    echo "[skip] $config not found" | tee -a $LOG
    continue
  fi
  echo "" | tee -a $LOG
  echo "=== Training LoRA: $genre at $(date) ===" | tee -a $LOG
  if uv run python training/lora_train.py --config "$config" >> "$LOG" 2>&1; then
    echo "  ✓ $genre completed at $(date)" | tee -a $LOG
  else
    echo "  ✗ $genre FAILED at $(date) (continuing queue)" | tee -a $LOG
  fi
done

echo "" | tee -a $LOG
echo "=== Queue done at $(date) ===" | tee -a $LOG
