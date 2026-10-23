#!/usr/bin/env python3
"""Full-precision companion to scripts/theory_validation.py.

The preregistration record (results/real/theory_validation.json) rounds
lambda/margin/tau at write time; the printed table in the paper inherits
those roundings, and raw L* is sensitive to them through the denominator
log(1/(1-lambda)) ~ lambda (e.g. Nowplaying raw L* shifts by ~5 between
lambda=0.0032 and the true 0.00316). This script re-measures the same
seeded quantities WITHOUT rounding and writes theory_fullprec.json, so
every raw L* in Table 4 is reproducible from stored digits.

Deterministic: same rng seeds and checkpoints as theory_validation.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from mplignn.data import load_sessions, pad_batch, build_adjacency  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402
import mplignn.model as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("tv", CODE / "scripts" / "theory_validation.py")
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)          # main() is guarded; no side effects

R = CODE / "results" / "real"

_real_entmax = M.entmax
CAP = []
def spy(z, alpha=1.5, dim=-1, n_iter=40):
    if z.dim() >= 2 and z.numel() < 4_000_000:
        CAP.append(z.detach())
    return _real_entmax(z, alpha, dim, n_iter)


def measure_margins(model, test_sessions):
    """Mean top1-top2 margin and EntMax support width at alpha=1.5."""
    model.eval(); CAP.clear()
    with torch.no_grad():
        for s in test_sessions[:256]:
            seqs, lens, _tgt = pad_batch([s])
            a = build_adjacency(seqs, lens)
            model(seqs, lens, a)
    margins, widths = [], []
    for z in CAP:
        zz = z.reshape(-1, z.shape[-1])
        for row in zz[:: max(1, zz.shape[0] // 200)]:
            top = row.topk(min(2, row.numel())).values
            margins.append(float(top[0] - top[-1]) if top.numel() > 1 else 0.0)
            w = _real_entmax(row, 1.5, -1)
            nz = row[w > 0]
            if nz.numel():
                widths.append(float(top[0]) - float(nz.min()))
    return float(np.mean(margins)), float(np.mean(widths))


def main():
    M.entmax = spy
    out = {"note": ("full-precision re-measurement of theory_validation.py "
                    "inputs; same seeds, same checkpoints, no rounding"),
           "datasets": {}}
    for ds, ck in tv.CKPT.items():
        data = load_sessions(CODE / tv.PKL[ds])
        lam = tv.spectral_gap(data.test)                      # full precision
        model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=1,
                         n_heads=2, global_blocks=2, vocab_blocks=1,
                         k_r=10, no_residual=True)
        model.load_state_dict(torch.load(R / ck, map_location="cpu",
                                         weights_only=True))
        m0, tau_w = measure_margins(model, data.test)
        lam_eff = min(max(lam, 1e-4), 0.999)
        Lstar = 1 + float(np.log(max(m0 / max(tau_w, 1e-9), 1e-9))
                          / np.log(1 / (1 - lam_eff)))
        out["datasets"][ds] = {"lambda": lam, "margin0_top1_top2": m0,
                               "tau_support_width": tau_w, "raw_L*": Lstar}
        print(f"{ds:18s} lambda={lam:.7f} margin={m0:.6f} "
              f"tau={tau_w:.6f} rawL*={Lstar:.2f}")
    (R / "theory_fullprec.json").write_text(json.dumps(out, indent=2))
    print("wrote", R / "theory_fullprec.json")


if __name__ == "__main__":
    main()
