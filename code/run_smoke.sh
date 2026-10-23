#!/usr/bin/env bash
# End-to-end code validation on the synthetic benchmark.
#
# Runs: unit tests -> data generation -> pilot study -> MPLI-GNN, SR-GNN
# baseline and two ablations -> noise injection -> shift stratification ->
# efficiency benchmark.  Produces results/smoke_report.md.
#
# NOTE: synthetic numbers validate the *code and mechanisms only*; they are
# NOT the paper's benchmark results and must never be pasted into the
# manuscript tables.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
EPOCHS=${EPOCHS:-8}
mkdir -p results data

echo "=== 1/7 unit tests ==="
$PY tests/test_model.py

echo "=== 2/7 synthetic data ==="
$PY scripts/make_synthetic.py --out data/synthetic.pkl

echo "=== 3/7 pilot study ==="
$PY scripts/pilot_study.py --data data/synthetic.pkl --out results/pilot.json

echo "=== 4/7 train MPLI-GNN ==="
$PY scripts/train.py --data data/synthetic.pkl --model mplignn \
    --epochs $EPOCHS --out results/mplignn.json

echo "=== 5/7 train SR-GNN baseline + ablations ==="
$PY scripts/train.py --data data/synthetic.pkl --model srgnn \
    --epochs $EPOCHS --out results/srgnn.json
$PY scripts/train.py --data data/synthetic.pkl --model mplignn \
    --ablation no_denoise --epochs $EPOCHS --out results/abl_no_denoise.json
$PY scripts/train.py --data data/synthetic.pkl --model mplignn \
    --ablation softmax_attn --epochs $EPOCHS --out results/abl_softmax.json

echo "=== 6/7 noise injection + shift stratification ==="
$PY scripts/noise_injection.py --data data/synthetic.pkl \
    --ckpt results/mplignn.pt --out results/noise.json
$PY scripts/stratify_shift.py --data data/synthetic.pkl \
    --ckpt results/mplignn.pt --out results/stratify.json

echo "=== 7/7 efficiency ==="
$PY scripts/efficiency.py --data data/synthetic.pkl --out results/efficiency.json

echo "smoke pipeline finished; see results/"
