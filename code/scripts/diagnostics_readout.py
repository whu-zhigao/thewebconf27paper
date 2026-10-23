#!/usr/bin/env python3
"""Read-out diagnostics for the revision (Section 5.6).

Reports, for a trained checkpoint on a given dataset:
  * the per-session entropy of the intent weights beta
  * gate sparsity at both EntMax attention sites (fraction of exact zeros)
  * the mean adaptive temperature alpha at each site
and, as a first-order answer to "is the head local-primary", the correlation
between beta entropy (normalised by log t) and correctness of the Top-20 hit.

Usage (paths in the paper tree):
  python diagnostics_readout.py --data <path>/data/diginetica.pkl \
      --ckpt <path>/results/real/abl_no_residual_diginetica.pt \
      --local-layers 1 --no-residual --out <path>/results/rev/readout_diginetica.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

from mplignn.data import (FastRelationIndex, load_sessions, pad_batch,  # noqa: E402
                          build_adjacency)
from mplignn.metrics import full_sort_metrics  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-layers", type=int, default=1)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--k-r", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    data = load_sessions(args.data)
    stem = args.data.stem
    model = MPLI_GNN(n_items=data.n_items, d=64,
                     local_layers=args.local_layers,
                     n_heads=2, global_blocks=2, vocab_blocks=1,
                     k_r=args.k_r,
                     no_residual=args.no_residual)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    rel = FastRelationIndex(data.test, data.train, data.n_items, args.k_r,
                            cache=args.data.parent / "relcache" / f"{stem}_test.npz")

    ent, ent_norm, correct, diag_acc = [], [], [], defaultdict(list)
    for i in range(0, len(data.test), args.batch_size):
        chunk = data.test[i:i + args.batch_size]
        seqs, lens, targets = pad_batch(chunk)
        a = build_adjacency(seqs, lens)
        ids = torch.arange(i, i + len(chunk))
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        out = model(seqs, lens, a, a_r, r0)
        beta = out["beta"]                                    # (B, T)
        for b in range(len(chunk)):
            t = int(lens[b])
            p = beta[b, :t].clamp_min(1e-12)
            h = float(-(p * p.log()).sum())
            ent.append(h)
            ent_norm.append(h / np.log(max(t, 2)))
        top = out["scores"].topk(20, dim=-1).indices
        correct.extend((top == targets.view(-1, 1)).any(1).tolist())
        for k, v in out["diag"].items():
            diag_acc[k].append(v)

    ent = np.array(ent)
    ent_norm = np.array(ent_norm)
    correct = np.array(correct, dtype=float)
    res = {
        "dataset": stem,
        "checkpoint": str(args.ckpt),
        "n_test": int(len(ent)),
        "beta_entropy_mean": float(ent.mean()),
        "beta_entropy_normalised_mean": float(ent_norm.mean()),
        "gate": {k: float(np.mean(v)) for k, v in diag_acc.items()},
        "hr20": float(correct.mean()),
        "corr_entropy_norm_vs_correct": float(np.corrcoef(ent_norm, correct)[0, 1]),
        "hr20_low_entropy_half": float(correct[ent_norm <= np.median(ent_norm)].mean()),
        "hr20_high_entropy_half": float(correct[ent_norm > np.median(ent_norm)].mean()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
