#!/usr/bin/env bash
# Local CPU patch: fifth seed s5 (12 configs) + sensitivity sweeps (bash 3.2 safe).
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
mkdir -p results/real

extra_for() {
  case "$1" in
    mplignn_L1) echo "--ablation no_residual" ;;
    *) echo "" ;;
  esac
}
model_for() {
  case "$1" in
    mplignn_L1|mplignn) echo mplignn ;;
    *) echo "$1" ;;
  esac
}

echo "=== part 1: fifth seed (12 configs) ==="
for m in mplignn_L1 mplignn srgnn narm; do
  for d in diginetica gowalla yoochoose_quarter; do
    out="results/real/seed_${m}_${d}_s5.json"
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo ">>> seed_s5 $m/$d $(date +%H:%M)"
    if $PY scripts/train.py --data "data/$d.pkl" --model "$(model_for $m)" $(extra_for $m) \
        --lr 5e-4 --epochs 12 --patience 4 --max-batches 300 --seed 5 \
        --out "$out" >> "results/real/log_s5_${m}_${d}.txt" 2>&1; then
      echo "    OK"
    else
      echo "    FAIL (see log_s5_${m}_${d}.txt)"
    fi
  done
done

echo "=== part 2: sensitivity sweeps ==="
for d in diginetica nowplaying; do
  out="results/real/sensitivity_${d}.json"
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo ">>> sensitivity $d $(date +%H:%M)"
  if $PY scripts/sensitivity.py --data "data/${d}.pkl" --epochs 12 \
      --out "$out" >> "results/real/log_sens_${d}.txt" 2>&1; then
    echo "    OK"
  else
    echo "    FAIL (see log_sens_${d}.txt)"
  fi
done
echo "LOCAL PATCH DONE"
