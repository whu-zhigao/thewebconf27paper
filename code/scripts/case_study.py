#!/usr/bin/env python3
"""Case study: 2 drift sessions; Top-5 lists of MPLI(L1) vs NARM vs truth."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mplignn.data import load_sessions, pad_batch, build_adjacency
from mplignn.baselines import BASELINES
from mplignn.model import MPLI_GNN
from scripts.pilot_study import cooccurrence_embeddings

OUT = Path("results/real"); data = load_sessions(Path("data/diginetica.pkl"))
emb = cooccurrence_embeddings(data.train, data.n_items)
def drift(s):
    k = max(1, len(s)//3); a = emb[s[:k]].mean(0); b = emb[s[-k:]].mean(0)
    return 1-float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
dv = np.array([drift(s) for s in data.test]); thr = np.percentile(dv, 75)
cand = [int(i) for i in np.argsort(-dv) if 4 <= len(data.test[i]) <= 8][:60]
sel = [cand[0], cand[len(cand)//2]]

mpli = MPLI_GNN(n_items=data.n_items, d=64, local_layers=1, n_heads=2,
                global_blocks=2, vocab_blocks=1, k_r=10, no_residual=True)
mpli.load_state_dict(torch.load(OUT/"abl_no_residual_diginetica.pt", map_location="cpu", weights_only=True))
mpli.eval()
narm = BASELINES["narm"](n_items=data.n_items, d=64)
narm.load_state_dict(torch.load(OUT/"narm_diginetica.pt", map_location="cpu", weights_only=True))
narm.eval()

cases = []
with torch.no_grad():
    for ci in sel:
        s = data.test[ci]; prefix = s[:-1]; gold = int(s[-1])
        seqs, lens, _t = pad_batch([prefix]); a = build_adjacency(seqs, lens)
        sm = mpli(seqs, lens, a); sm = sm["scores"] if isinstance(sm, dict) else sm
        sn = narm(seqs, lens, a);  sn = sn["scores"] if isinstance(sn, dict) else sn
        tm = sm[0].argsort(descending=True)[:5].tolist()
        tn = sn[0].argsort(descending=True)[:5].tolist()
        cases.append({"index": ci, "length": len(s), "drift": float(dv(s)),
                      "session_items": [int(x) for x in prefix],
                      "gold": gold,
                      "mpli_L1_top5": [int(x) for x in tm],
                      "mpli_L1_rank": int((sm[0].argsort(descending=True) == gold).nonzero()[0])+1 if gold in sm[0].argsort(descending=True).tolist() else -1,
                      "narm_top5": [int(x) for x in tn],
                      "narm_rank": int((sn[0].argsort(descending=True) == gold).nonzero()[0])+1 if gold in sn[0].argsort(descending=True).tolist() else -1})
(OUT/"case_study_diginetica.json").write_text(json.dumps(
    {"threshold": float(thr), "cases": cases}, indent=2))
print(json.dumps(cases, indent=2)[:800])
