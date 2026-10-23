#!/usr/bin/env python3
"""Cross-architecture check of the calibration rule's PRECONDITION.

Rule (decision form): enable global evidence only if the architecture has
a support-based gate AND the attention margin exceeds the support width.
GCE-GNN uses dense tanh attention (no support), so the rule predicts no
global benefit -- exactly what the audit found. This script measures the
precondition on both architectures' checkpoints: exact-zero sparsity of
the gating weights and the margin/support-width ratio where defined."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mplignn.model as M
import mplignn.baselines as B
from mplignn.data import load_sessions, pad_batch, build_adjacency
from mplignn.model import MPLI_GNN
from mplignn.entmax import entmax as _real_entmax
from scripts.theory_validation import spectral_gap

P = Path(__file__).resolve().parents[1]; R = P/"results"/"real"
CAP = []
_orig = M.entmax
def spy(z, alpha=1.5, dim=-1, n_iter=40):
    if z.dim() >= 2 and z.numel() < 4_000_000: CAP.append(z.detach())
    return _orig(z, alpha, dim, n_iter)

GLOB = []
_orig_attend = B.GlobalEncoder._attend
def attend_spy(self, x):
    q = self.wq(x).view(x.shape[0], self.h, self.dk).transpose(0,1)
    k = self.wk(x).view(x.shape[0], self.h, self.dk).transpose(0,1)
    v = self.wv(x).view(x.shape[0], self.h, self.dk).transpose(0,1)
    s = torch.tanh(q @ k.transpose(-2,-1) / self.dk**0.5)
    GLOB.append(s.detach())
    out = s @ v
    out = out.transpose(0,1).reshape(x.shape[0], self.d)
    return self.drop(self.wo(out))

CKPT = {"diginetica": "diginetica", "gowalla": "gowalla",
        "yoochoose_quarter": "yoochoose_quarter"}
out = {}
for ds in CKPT:
    data = load_sessions(P/"data"/f"{ds}.pkl")
    # MPLI L1: entmax gate stats
    M.entmax = spy; CAP.clear()
    m = MPLI_GNN(n_items=data.n_items, d=64, local_layers=1, n_heads=2,
                 global_blocks=2, vocab_blocks=1, k_r=10, no_residual=True)
    ck = "abl_no_residual_diginetica.pt" if ds=="diginetica" else f"mplignn_L1_{ds}.pt"
    m.load_state_dict(torch.load(R/ck, map_location="cpu", weights_only=True)); m.eval()
    with torch.no_grad():
        for s in data.test[:128]:
            seqs, lens, _ = pad_batch([s]); a = build_adjacency(seqs, lens)
            m(seqs, lens, a)
    zeros, marg = [], []
    for z in CAP:
        w = _orig(z, 1.5, -1)
        zeros.append(float((w == 0).float().mean()))
        zz = z.reshape(-1, z.shape[-1])
        for row in zz[::max(1, zz.shape[0]//100)]:
            t = row.topk(min(2, row.numel())).values
            marg.append(float(t[0]-t[-1]))
    # GCE-GNN: dense tanh gate stats
    B.GlobalEncoder._attend = attend_spy; GLOB.clear()
    g = B.BASELINES["gcegnn"](n_items=data.n_items, d=64, use_global=True)
    g.load_state_dict(torch.load(R/f"gcegnn_{ds}.pt", map_location="cpu", weights_only=True)); g.eval()
    with torch.no_grad():
        for s in data.test[:128]:
            seqs, lens, _ = pad_batch([s]); a = build_adjacency(seqs, lens)
            g(seqs, lens, a)
    gz, gm = [], []
    for s_ in GLOB:
        gz.append(float((s_ == 0).float().mean()))
        ss = s_.reshape(-1, s_.shape[-1])
        for row in ss[::max(1, ss.shape[0]//100)]:
            t = row.topk(min(2, row.numel())).values
            gm.append(float(t[0]-t[-1]))
    M.entmax = _orig; B.GlobalEncoder._attend = _orig_attend
    out[ds] = {
      "mpli_gate": {"exact_zero_frac": round(float(np.mean(zeros)), 4),
                    "mean_margin": round(float(np.mean(marg)), 4)},
      "gcegnn_gate": {"exact_zero_frac": round(float(np.mean(gz)), 4),
                      "mean_margin": round(float(np.mean(gm)), 4)}}
    print(ds, out[ds])
(R/"calibration_transfer.json").write_text(json.dumps(out, indent=2))
print("wrote calibration_transfer.json")
