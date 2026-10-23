#!/usr/bin/env python3
"""RQ5 noise-injection robustness (Section 5.7.1).

Loads a checkpoint produced by train.py and reports HR@20 / MRR@20 under
random-item corruption ratios x in {0,10,...,50}%.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import FastRelationIndex, load_sessions
from mplignn.model import MPLI_GNN, SRGNNLite
from mplignn.baselines import BASELINES
from scripts.train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--model", default="mplignn",
                    choices=["mplignn", "srgnn"] + list(BASELINES))
    ap.add_argument("--ablation", default="")
    ap.add_argument("--k-r", default=10, type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = load_sessions(args.data)
    flags = {f.strip(): True for f in args.ablation.split(",") if f.strip()}
    if args.model in BASELINES:
        model = BASELINES[args.model](n_items=data.n_items, d=args.dim)
    elif args.model == "srgnn":
        model = SRGNNLite(n_items=data.n_items, d=args.dim)
    else:
        model = MPLI_GNN(n_items=data.n_items, d=args.dim,
                         k_r=args.k_r, **flags)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    cache = args.data.parent / "relcache" / f"{args.data.stem}_test.npz"
    rel = FastRelationIndex(data.test, data.train, data.n_items, args.k_r,
                            cache=cache)

    table = {}
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        rng = np.random.default_rng(100)
        m = evaluate(model, data.test, rel, "cpu", noise_ratio=x, rng=rng)
        table[f"{int(x * 100)}%"] = {"HR@20": round(m["HR@20"], 4),
                                     "MRR@20": round(m["MRR@20"], 4)}
        print(f"x={int(x * 100):2d}%  HR@20={m['HR@20']:.4f}  "
              f"MRR@20={m['MRR@20']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
