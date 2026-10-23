#!/usr/bin/env bash
# Resumable baseline matrix: 11 baselines x 6 datasets under the shared
# CPU budget. Skips any run whose output JSON already exists.
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
EPOCHS=12; PATIENCE=4; MAXB=300; LR=5e-4
MODELS="fpmc gru4rec narm stamp dsan lessr gcegnn mtd msgat dhcn msgifsr"
DATASETS="diginetica yoochoose yoochoose_quarter gowalla nowplaying retailrocket"
mkdir -p results/real

total=0; done_ct=0; skip_ct=0; fail_ct=0
for d in $DATASETS; do
  for m in $MODELS; do
    out="results/real/${m}_${d}.json"
    total=$((total+1))
    if [ -s "$out" ]; then skip_ct=$((skip_ct+1)); continue; fi
    echo ">>> $m / $d  ($(date +%H:%M:%S))"
    if $PY scripts/train.py --data "data/$d.pkl" --model "$m" \
        --lr $LR --epochs $EPOCHS --patience $PATIENCE --max-batches $MAXB \
        --out "$out" >> "results/real/log_${m}_${d}.txt" 2>&1; then
      done_ct=$((done_ct+1))
      $PY -c "import json;t=json.load(open('$out'))['test'];print('    HR@20 %.2f  MRR@20 %.2f'%(100*t['HR@20'],100*t['MRR@20']))"
    else
      fail_ct=$((fail_ct+1)); echo "    FAILED (see results/real/log_${m}_${d}.txt)"
    fi
  done
done
echo "MATRIX DONE: ok=$done_ct skip=$skip_ct fail=$fail_ct total=$total"
