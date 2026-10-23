#!/usr/bin/env python3
"""RQ5 shift-stratified evaluation (Section 5.7.2).

Splits test sessions into shift / stable strata using the embedding-drift
detector of the pilot study (or the planted oracle labels when present)
and reports metrics per stratum.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import FastRelationIndex, load_sessions, pad_batch, build_adjacency
from mplignn.metrics import full_sort_metrics
from mplignn.model import MPLI_GNN, SRGNNLite
from scripts.pilot_study import cooccurrence_embeddings


@torch.no_grad()
def eval_subset(model, sessions, rel, ids, device="cpu"):
    agg, n = {}, 0
    for i in range(0, len(sessions), 128):
        chunk = sessions[i:i + 128]
        a_r, r0 = rel.batch_relation(ids[i:i + len(chunk)],
                                     model.embedding.weight)
        seqs, lens, targets = pad_batch(chunk, device)
        a = build_adjacency(seqs, lens)
        out = model(seqs, lens, a, a_r, r0)
        for k, v in full_sort_metrics(out["scores"], targets).items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--ablation", default="")
    ap.add_argument("--k-r", default=10, type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = load_sessions(args.data)
    emb = cooccurrence_embeddings(data.train, data.n_items)
    drifts = []
    for s in data.test:
        t3 = max(1, len(s) // 3)
        drifts.append(float(emb[s[:t3]].mean(0) @ emb[s[-t3:]].mean(0)))
    detected = np.array(drifts) < np.percentile(drifts, 25)
    oracle = data.shift_flags if data.shift_flags is not None else detected

    flags = {f.strip(): True for f in args.ablation.split(",") if f.strip()}
    model = MPLI_GNN(n_items=data.n_items, d=args.dim, k_r=args.k_r, **flags)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()
    cache = args.data.parent / "relcache" / f"{args.data.stem}_test.npz"
    rel = FastRelationIndex(data.test, data.train, data.n_items, args.k_r,
                            cache=cache)

    shift_sessions = [s for s, f in zip(data.test, oracle) if f]
    stable_sessions = [s for s, f in zip(data.test, oracle) if not f]
    shift_ids = torch.nonzero(torch.from_numpy(oracle)).flatten()
    stable_ids = torch.nonzero(torch.from_numpy(~oracle)).flatten()
    out = {
        "detector": "oracle" if data.shift_flags is not None else "drift",
        "n_shift": len(shift_sessions), "n_stable": len(stable_sessions),
        "shift": eval_subset(model, shift_sessions, rel, shift_ids),
        "stable": eval_subset(model, stable_sessions, rel, stable_ids),
    }
    out["gain_on_shift_minus_stable"] = {
        k: round(out["shift"][k] - out["stable"][k], 4)
        for k in out["shift"]}
    print(json.dumps(out, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
