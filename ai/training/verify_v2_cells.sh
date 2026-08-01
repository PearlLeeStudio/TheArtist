#!/usr/bin/env bash
# Verify v2 chain cells — checks each expected cell has the artifacts
# it should have for the configured method, and reports a one-line
# status per cell. Safe to run repeatedly during training.
#
# Usage:
#   bash ai/training/verify_v2_cells.sh [scope]
#     scope: a | b2 | b3 | all   (default: all)
#
# Exit code: 0 if all expected cells passed, 1 otherwise.

set -uo pipefail
cd "$(dirname "$0")/.."

SCOPE="${1:-all}"

GENRES_ALL=(blues bossa classical country electronic folk funk gospel hip_hop rnb_soul rock)

check_full_cell() {
  local rn="$1"; local label="$2"
  if [[ -s "checkpoints/${rn}/best.pt" ]]; then
    local epoch
    epoch=$(grep -oE "Epoch +[0-9]+/8" "logs/${rn}_"*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/8" | head -1)
    echo "  ✓ ${label}: best.pt ok (last epoch ${epoch:-?})"
    return 0
  fi
  echo "  ✗ ${label}: best.pt MISSING"
  return 1
}

check_lora_cell() {
  local rn="$1"; local label="$2"
  if [[ -s "checkpoints/${rn}/adapter/adapter_config.json" && -s "checkpoints/${rn}/embedding_extension.pt" ]]; then
    local epoch
    epoch=$(grep -oE "Epoch +[0-9]+/8" "logs/${rn}_"*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/8" | head -1)
    echo "  ✓ ${label}: adapter+embedding ok (last epoch ${epoch:-?})"
    return 0
  fi
  echo "  ✗ ${label}: adapter+embedding MISSING"
  return 1
}

check_prefix_cell() {
  local rn="$1"; local label="$2"
  if [[ -s "checkpoints/${rn}/prefix_state.pt" && -s "checkpoints/${rn}/embedding_extension.pt" ]]; then
    local epoch
    epoch=$(grep -oE "Epoch +[0-9]+/8" "logs/${rn}_"*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/8" | head -1)
    echo "  ✓ ${label}: prefix+embedding ok (last epoch ${epoch:-?})"
    return 0
  fi
  echo "  ✗ ${label}: prefix+embedding MISSING"
  return 1
}

pass=0; fail=0
tally() { if [[ "$1" -eq 0 ]]; then pass=$((pass+1)); else fail=$((fail+1)); fi; }

if [[ "$SCOPE" == "a" || "$SCOPE" == "all" ]]; then
  echo "=== A2: BitFit/IA3/Full FT × 11 genres (33 cells, seed42) ==="
  for g in "${GENRES_ALL[@]}"; do
    check_full_cell "ft_f1_bitfit_${g}"  "bitfit/${g}";  tally $?
    check_lora_cell "ft_f1_ia3_${g}"     "ia3/${g}";     tally $?
    check_full_cell "ft_f1_full_ft_${g}" "full_ft/${g}"; tally $?
  done
  echo "=== A3: Prefix × 11 genres (11 cells, seed42) ==="
  for g in "${GENRES_ALL[@]}"; do
    check_prefix_cell "ft_f1_prefix_${g}" "prefix/${g}"; tally $?
  done
fi

if [[ "$SCOPE" == "b2" || "$SCOPE" == "all" ]]; then
  echo "=== B2: 4 methods × 11 genres × seed {123, 7} = 88 cells ==="
  for g in "${GENRES_ALL[@]}"; do
    for s in 123 7; do
      check_full_cell "ft_f1_bitfit_${g}_seed${s}"  "bitfit/${g}/seed${s}";  tally $?
      check_lora_cell "ft_f1_ia3_${g}_seed${s}"     "ia3/${g}/seed${s}";     tally $?
      check_full_cell "ft_f1_full_ft_${g}_seed${s}" "full_ft/${g}/seed${s}"; tally $?
      check_prefix_cell "ft_f1_prefix_${g}_seed${s}" "prefix/${g}/seed${s}"; tally $?
    done
  done
fi

if [[ "$SCOPE" == "b3" || "$SCOPE" == "all" ]]; then
  echo "=== B3_lora: 11 genres × seed {42, 123, 7} = 33 cells ==="
  declare -A SELECTED_RANK=(
    [blues]=r32 [bossa]=r16 [classical]=r64 [country]=r4 [electronic]="" \
    [folk]=r32 [funk]=r64 [gospel]=r4 [hip_hop]=r4 [rnb_soul]=r4 [rock]=r16
  )
  for g in "${GENRES_ALL[@]}"; do
    local_suffix="${SELECTED_RANK[$g]}"
    if [[ -n "$local_suffix" ]]; then
      check_lora_cell "ft_f1_lora_${g}_${local_suffix}" "lora/${g}/seed42";        tally $?
    else
      check_lora_cell "ft_f1_lora_${g}"                  "lora/${g}/seed42";        tally $?
    fi
    check_lora_cell "ft_f1_lora_${g}_${local_suffix:+${local_suffix}_}seed123"     "lora/${g}/seed123"; tally $?
    check_lora_cell "ft_f1_lora_${g}_${local_suffix:+${local_suffix}_}seed7"       "lora/${g}/seed7";   tally $?
  done
fi

echo ""
echo "=== SUMMARY ==="
echo "  PASS: $pass"
echo "  FAIL: $fail"
if [[ "$fail" -gt 0 ]]; then exit 1; fi
exit 0
