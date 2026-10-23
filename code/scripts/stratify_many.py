#!/usr/bin/env python3
"""Shift-stratified evaluation for baseline checkpoints (Diginetica)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mplignn.data import load_sessions, pad_batch, build_adjacency
from mplignn.metrics import full_sort_metrics
from mplignn.baselines import BASELINES
from mplignn.model import MPLI_GNN, SRGNNLite
from scripts.pilot_study import cooccurrence_embeddings

OUT = Path("results/real"); DATA = Path("data/diginetica.pkl")
data = load_sessions(DATA)

# drift split identical to pilot P2: cosine drift of first/last thirds, 75th pct
emb = cooccurrence_embeddings(data.train, data.n_items)
def drift(s):
    k = max(1, len(s)//3)
    a = emb[s[:k]].mean(0); b = emb[s[-k:]].mean(0)
    c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)); return 1-c
dv = np.array([drift(s) for s in data.test]); thr = np.percentile(dv, 75)
shift = dv >= thr

MODELS = {
 "srgnn": (lambda: SRGNNLite(n_items=data.n_items, d=64, dropout=0.1), OUT/"srgnn_diginetica.pt"),
 "narm":  (lambda: BASELINES["narm"](n_items=data.n_items, d=64), OUT/"narm_diginetica.pt"),
 "gcegnn":(lambda: BASELINES["gcegnn"](n_items=data.n_items, d=64), OUT/"gcegnn_diginetica.pt"),
 "dsan":  (lambda: BASELINES["dsan"](n_items=data.n_items, d=64), OUT/"dsan_diginetica.pt"),
 "msgat": (lambda: BASELINES["msgat"](n_items=data.n_items, d=64), OUT/"msgat_diginetica.pt"),
 "mplignn_L1": (lambda: MPLI_GNN(n_items=data.n_items, d=64, local_layers=1,
                  n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                  no_residual=True), OUT/"abl_no_residual_diginetica.pt"),
 "mplignn": (lambda: MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10), OUT/"mplignn_diginetica.pt"),
}
res = {"threshold": float(thr), "n_shift": int(shift.sum()), "n_stable": int((~shift).sum())}
for name, (mk, ck) in MODELS.items():
    if not ck.exists(): print("missing", ck); continue
    m = mk(); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); m.eval()
    out = {}
    with torch.no_grad():
        for lab, sel in (("shift", shift), ("stable", ~shift)):
            agg, n = {}, 0
            idx = np.where(sel)[0]
            for i in range(0, len(idx), 128):
                chunk = [data.test[j] for j in idx[i:i+128]]
                seqs, lens, _targets = pad_batch(chunk)
                a = build_adjacency(seqs, lens)
                r = m(seqs, lens, a)
                sc = r["scores"] if isinstance(r, dict) else r
                for bi, s in enumerate(chunk):
                    t = full_sort_metrics(sc[bi], s[-1])
                    for k, v in t.items(): agg[k] = agg.get(k, 0)+v
                n += len(chunk)
            out[lab] = {k: v/n for k, v in agg.items()}
    res[name] = out
    print(name, {k: round(100*v["HR@20"],2) for k, v in out.items()})
(OUT/"stratified_diginetica.json").write_text(json.dumps(res, indent=2))
print("wrote stratified_diginetica.json")
