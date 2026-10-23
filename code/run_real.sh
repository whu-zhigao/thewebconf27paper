#!/usr/bin/env bash
# Full real-data experiment suite for the paper (Sections 5.1-5.7).
#
# Runs on CPU with a documented compute budget: d=64, 8 epochs, up to 300
# batches per epoch, early stopping with patience 3.  These settings are
# reported honestly in the paper's implementation details.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
DATASETS="diginetica gowalla yoochoose nowplaying retailrocket"
EPOCHS=${EPOCHS:-8}
MAXB=${MAXB:-300}
mkdir -p results/real

echo "=== Phase A: pilot studies + Item-KNN ==="
for d in $DATASETS; do
  $PY scripts/pilot_study.py --data data/$d.pkl --out results/real/pilot_$d.json
  $PY scripts/train.py --data data/$d.pkl --model itemknn \
      --out results/real/itemknn_$d.json
done

echo "=== Phase B: SR-GNN-lite baseline ==="
for d in $DATASETS; do
  $PY scripts/train.py --data data/$d.pkl --model srgnn \
      --epochs $EPOCHS --max-batches $MAXB \
      --out results/real/srgnn_$d.json
done

echo "=== Phase C: MPLI-GNN ==="
for d in $DATASETS; do
  $PY scripts/train.py --data data/$d.pkl --model mplignn \
      --epochs $EPOCHS --max-batches $MAXB \
      --out results/real/mplignn_$d.json
done

echo "=== Phase D: ablations on Diginetica ==="
for abl in no_denoise softmax_attn no_anchor no_global; do
  $PY scripts/train.py --data data/diginetica.pkl --model mplignn \
      --ablation $abl --epochs $EPOCHS --max-batches $MAXB \
      --out results/real/abl_${abl}_diginetica.json
done

echo "=== Phase E: robustness + stratification + efficiency ==="
$PY scripts/noise_injection.py --data data/diginetica.pkl \
    --ckpt results/real/mplignn_diginetica.pt \
    --out results/real/noise_diginetica.json
$PY scripts/stratify_shift.py --data data/diginetica.pkl \
    --ckpt results/real/mplignn_diginetica.pt \
    --out results/real/stratify_diginetica.json
$PY scripts/noise_injection.py --data data/diginetica.pkl \
    --ckpt results/real/srgnn_diginetica.pt \
    --out results/real/noise_srgnn_diginetica.json || true
$PY scripts/efficiency.py --data data/diginetica.pkl \
    --out results/real/efficiency_diginetica.json

echo "ALL REAL-DATA EXPERIMENTS DONE"
