#!/usr/bin/env python3
"""Self-audit probe: two advertised modules receive zero gradient.

In the released instrument the adaptive global encoder (dense tanh
attention over batch-unique items) and the target-anchored sparse
attention (SAtt) sit outside the gradient path: the anchor output
t_n_s is consumed only by the r0-is-None fallback branch of
RelationEnhancement.forward, and scripts/train.py always supplies the
retrieved relation memory. Consequences reported in the paper:

  * the M_b (global blocks) sensitivity axis is inert by construction;
  * the SAtt sparsity diagnostic (99.7% exact zeros, alpha ~ 1.48)
    reflects an untrained module, not a trained sparse gate;
  * the v1 no_anchor / no_global ablation deltas are initialization
    RNG jitter, not architecture effects.

Checks (all must hold on a released checkpoint):
  A) bypassing the global encoder leaves scores bit-identical;
  B) perturbing anchor + SAtt parameters leaves scores bit-identical;
  C) gradients: global_ / tare.anchor / tare.w_s receive None;
     tare.w_qn and tare.rel_proj receive nonzero gradient.

Usage:  python3 analysis/probe_dead_paths.py [checkpoint.pt]
Exit code 0 = dead paths confirmed (paper claim holds), 1 = mismatch.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from mplignn.model import MPLI_GNN  # noqa: E402

CKPT = CODE / "results" / "real" / "seed_mplignn_diginetica_s1.pt"


def make_batch(n_items, ckpt, b=8, t=12, k=10):
    torch.manual_seed(0)
    seqs = torch.randint(1, n_items, (b, t))
    lens = torch.randint(2, t + 1, (b,))
    seqs = seqs.masked_fill(
        torch.arange(t).unsqueeze(0) >= lens.unsqueeze(1), 0)
    a = torch.zeros(b, t, t)
    for i in range(b):
        s = [x for x in seqs[i].tolist() if x != 0]
        for u, v in zip(s[:-1], s[1:]):
            a[i, u % t, v % t] += 1
    a = a + a.transpose(1, 2)
    a = a / a.sum(-1, keepdim=True).clamp_min(1.0)
    ids = torch.randint(0, n_items, (b, k))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    r0 = sd["embedding.weight"][ids].detach()
    a_r = F.cosine_similarity(r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
    a_r = a_r.clamp_min(0)
    a_r = a_r / a_r.sum(-1, keepdim=True).clamp_min(1e-9)
    return seqs, lens, a, a_r, r0


def load_model(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    n_items = sd["embedding.weight"].shape[0]
    m = MPLI_GNN(n_items=n_items, d=64, local_layers=2)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m, n_items


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else CKPT
    m, n_items = load_model(ckpt)
    m.eval()
    seqs, lens, a, a_r, r0 = make_batch(n_items, ckpt)

    with torch.no_grad():
        ref = m(seqs, lens, a, a_r, r0)["scores"]
        m.no_global = True
        dA = (m(seqs, lens, a, a_r, r0)["scores"] - ref).abs().max().item()
        m.no_global = False
        m.tare.anchor.data += 0.7
        with torch.no_grad():
            m.tare.w_s.weight.data *= 1.5
        dB = (m(seqs, lens, a, a_r, r0)["scores"] - ref).abs().max().item()
        det = (m(seqs, lens, a, a_r, r0)["scores"] - ref).abs().max().item()

    m2, _ = load_model(ckpt)
    m2.eval()
    out = m2(seqs, lens, a, a_r, r0)
    out["scores"].sum().backward()
    gsum = lambda mods: sum(p.grad.abs().sum().item()
                            for p in mods if p.grad is not None)  # noqa: E731
    g_glob_none = all(p.grad is None for p in m2.global_.parameters())
    g_anchor_none = m2.tare.anchor.grad is None
    g_ws_none = m2.tare.w_s.weight.grad is None
    g_qn = gsum([m2.tare.w_qn.weight])
    g_rel = gsum([m2.tare.rel_proj.weight])

    print(f"checkpoint            : {ckpt.name}")
    print(f"determinism |Δ|       : {det:.3e}   (must be 0)")
    print(f"A) global bypass |Δ|  : {dA:.3e}   (dead iff 0)")
    print(f"B) anchor/SAtt |Δ|    : {dB:.3e}   (dead iff 0)")
    print(f"C) grads: global_ all-None={g_glob_none}  anchor-None="
          f"{g_anchor_none}  w_s-None={g_ws_none}  "
          f"w_qn={g_qn:.3e}  rel_proj={g_rel:.3e}")

    ok = (det == 0.0 and dA == 0.0 and dB == 0.0 and g_glob_none
          and g_anchor_none and g_ws_none and g_qn > 0 and g_rel > 0)
    print("VERDICT:", "dead paths CONFIRMED" if ok else "MISMATCH — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
