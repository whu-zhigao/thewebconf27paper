#!/usr/bin/env python3
"""Operationalize Proposition 1 (P6): measure its three inputs and
REGISTER predicted L* per dataset BEFORE the cross-architecture audit is
unblinded (preregistration). Empirical L* is merged later from
audit_summary.json by b2_integrate.py.

Inputs:
  lambda      : mean spectral gap 1-|lambda2| of row-stochastic session
                graphs (data-only, no training).
  tau(alpha)  : mean empirical EntMax breakpoint = smallest pre-entmax
                score receiving nonzero weight (captured from the trained
                L=1 checkpoint by monkey-patching entmax).
  margin(0)   : mean top1-top2 gap of pre-entmax attention rows (query
                discriminability of the trained L=1 model).
Predicted L* = 1 + log(margin/tau) / log(1/(1-lambda)), clipped to [1,4].
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mplignn.model as M
from mplignn.data import load_sessions, pad_batch, build_adjacency
from mplignn.model import MPLI_GNN

R = Path("results/real")
CKPT = {"diginetica": "abl_no_residual_diginetica.pt",
        "gowalla": "mplignn_L1_gowalla.pt",
        "yoochoose_quarter": "mplignn_L1_yoochoose_quarter.pt",
        "nowplaying": "mplignn_L1_nowplaying.pt"}   # amendment, pre-unblinding
PKL = {"diginetica": "data/diginetica.pkl", "gowalla": "data/gowalla.pkl",
       "yoochoose_quarter": "data/yoochoose_quarter.pkl",
       "nowplaying": "data/nowplaying.pkl"}

def spectral_gap(sessions, cap=1500):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sessions), min(cap, len(sessions)), replace=False)
    gaps = []
    for i in idx:
        s = sessions[i]; n = len(set(s))
        if n < 3: continue
        A = np.zeros((n, n)); u = list(dict.fromkeys(s))
        pos = {v: k for k, v in enumerate(u)}
        for a, b in zip(s[:-1], s[1:]):
            if a != b: A[pos[a], pos[b]] += 1
        # BIDIRECTIONAL, matching the model's propagation operator
        # (SR-GNN convention concatenates reversed/out-going edges)
        A = A + A.T
        rs = A.sum(1, keepdims=True); rs[rs == 0] = 1; A = A / rs
        ev = np.abs(np.linalg.eigvals(A))
        gaps.append(1.0 - np.sort(ev)[-2])
    return float(np.mean(gaps))

CAP = []; ALPHAS = []
_real_entmax = M.entmax
def spy(z, alpha=1.5, dim=-1, n_iter=40):
    if z.dim() >= 2 and z.numel() < 4_000_000:
        CAP.append(z.detach())
        try: ALPHAS.append(float(alpha))
        except Exception: pass
    return _real_entmax(z, alpha, dim, n_iter)

def main():
    M.entmax = spy
    out = {"operator_note": ("lambda measured on the bidirectional row-normalized "
         "session adjacency, matching the model's propagation operator; "
         "amended before any GCE-GNN audit result existed"),
       "preregistered_predictions": {}}
    for ds, ck in CKPT.items():
        data = load_sessions(Path(PKL[ds]))
        lam = spectral_gap(data.test)
        model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=1,
                         n_heads=2, global_blocks=2, vocab_blocks=1,
                         k_r=10, no_residual=True)
        model.load_state_dict(torch.load(R/ck, map_location="cpu", weights_only=True))
        model.eval(); CAP.clear(); ALPHAS.clear()
        with torch.no_grad():
            for s in data.test[:256]:
                seqs, lens, _tgt = pad_batch([s])
                a = build_adjacency(seqs, lens)
                model(seqs, lens, a)
        margins, widths, top1s = [], [], []
        for z in CAP:
            zz = z.reshape(-1, z.shape[-1])
            for row in zz[:: max(1, zz.shape[0] // 200)]:
                top = row.topk(min(2, row.numel())).values
                margins.append(float(top[0] - top[-1]) if top.numel() > 1 else 0.0)
                top1s.append(float(top[0]))
                w = _real_entmax(row, 1.5, -1)
                nz = row[w > 0]
                if nz.numel():
                    widths.append(float(top[0]) - float(nz.min()))  # support width
        m0 = float(np.mean(margins))          # top1-top2 discriminability
        tau_w = float(np.mean(widths))        # entmax support width (same units)
        lam_eff = min(max(lam, 1e-4), 0.999)
        # benefit of global replacement active iff margin exceeds support width
        ratio = m0 / max(tau_w, 1e-9)
        Lstar = 1 + np.log(max(ratio, 1e-9)) / np.log(1 / (1 - lam_eff))
        out["preregistered_predictions"][ds] = {
            "adaptive_alpha_mean": round(float(np.mean(ALPHAS)), 3) if ALPHAS else None,
            "adaptive_alpha_range": [round(float(np.min(ALPHAS)), 3),
                                     round(float(np.max(ALPHAS)), 3)] if ALPHAS else None,
            "lambda": round(lam, 4),
            "margin0_top1_top2": round(m0, 4),
            "tau_support_width": round(tau_w, 4),
            "mean_top1": round(float(np.mean(top1s)), 4),
            "predicted_L*": int(np.clip(round(Lstar), 1, 4)),
            "raw_L*": round(float(Lstar), 2)}
        print(ds, out["preregistered_predictions"][ds])
    (R/"theory_validation.json").write_text(json.dumps(out, indent=2))
    print("wrote theory_validation.json (preregistered)")

if __name__ == "__main__":
    main()
