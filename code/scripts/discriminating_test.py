#!/usr/bin/env python3
"""Discriminating test for Proposition 1 (chair's BPA criterion).

Problem: on all real benchmarks the calibrated threshold predicts L*=1,
which cannot distinguish "gating at depth 1" from "no depth benefit
anywhere". This script CONSTRUCTS the regime where the theory risks
losing: a synthetic corpus of long, chain-like sessions (length ~28,
few repeats) whose session graphs mix slowly (small spectral gap),
so Eq. (L*) predicts L*>1 BEFORE the depth sweep is run.

Pipeline (prediction strictly before empirical unblinding):
  1. generate long-chain corpus with planted group structure
  2. measure lambda on test graphs  (data-only, exogenous)
  3. train MPLI-GNN L=1 on it, capture tau/margin via entmax spy
  4. predicted L* = 1 + log(margin/tau) / log(1/(1-lambda))
  5. ONLY THEN train full MPLI-GNN at L in {1,2,3} -> empirical L*
  6. write discriminating_test.json (timestamps prove the ordering)
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mplignn.model as M
from mplignn.data import load_sessions, pad_batch, build_adjacency, save_sessions, preprocess
from mplignn.model import MPLI_GNN
from scripts.theory_validation import spectral_gap, spy, _real_entmax

P = Path(__file__).resolve().parents[1]
OUT = P/"results"/"real"
LOG = open(P/"discriminating.log", "a", buffering=1)
def log(m): LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n"); print(m, flush=True)

N_ITEMS, N_GROUPS, N_TRAIN, N_TEST, LEN, SEED = 1200, 30, 10000, 600, 28, 7

def gen_corpus():
    rng = np.random.default_rng(SEED)
    groups = [rng.choice(N_ITEMS, N_ITEMS//N_GROUPS, replace=False) for _ in range(N_GROUPS)]
    sessions = []
    for i in range(N_TRAIN + N_TEST + 300):
        s, g = [], int(rng.integers(N_GROUPS))
        for _ in range(LEN):
            pool = groups[g]
            s.append(int(pool[rng.integers(len(pool))]))
            # PREDICTABLE cyclic group order (80%) with occasional random jump:
            # next-group is learnable, so a competent model reaches ~G-limited HR
            g = (g + 1) % N_GROUPS if rng.random() < 0.8 else int(rng.integers(N_GROUPS))
        sessions.append(s)
    tr = sessions[:N_TRAIN]
    va = sessions[N_TRAIN:N_TRAIN+300]
    te = sessions[N_TRAIN+300:]
    from mplignn.data import SessionData
    def mk(x): return SessionData(train=x, valid=va, test=te,
                                  n_items=N_ITEMS, shift_flags=None)
    return mk(tr)

def train(args, out):
    if (OUT/out).exists(): log(f"skip {out}"); return
    r = subprocess.run([sys.executable, "scripts/train.py", *map(str, args),
                        "--out", OUT/out], cwd=P, capture_output=True, text=True,
                       errors="replace")
    log(f"{out}: rc={r.returncode} ok={(OUT/out).exists()}")
    if not (OUT/out).exists():
        (OUT/f"fail_{Path(out).stem}.txt").write_text((r.stdout or "")[-3000:] + (r.stderr or "")[-3000:])

def main():
    t_start = time.time()
    log("DISCRIMINATING TEST start")
    data0 = gen_corpus()
    data = preprocess(data0)
    for f in P.glob("data/relcache/longchain_*.npz"):
        f.unlink()   # corpus regenerated: relation caches must not survive
    save_sessions(data, P/"data"/"longchain.pkl")
    log(f"corpus: {len(data0.train)}/{len(data0.valid)}/{len(data0.test)} -> filtered {len(data.train)}/{len(data.valid)}/{len(data.test)}, length {LEN}")

    lam = spectral_gap(data.test, cap=1200)
    log(f"lambda={lam:.4f} (expect << real benchmarks)")

    # train L=1 instrument, capture tau/margin
    M.entmax = spy
    train(["--data", "data/longchain.pkl", "--model", "mplignn",
           "--ablation", "no_residual", "--lr", "2e-3", "--epochs", "120",
           "--patience", "15", "--max-batches", "0", "--seed", "7",
           "--out", "dis_chain_L1.json"], "dis_chain_L1.json")
    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=1, n_heads=2,
                     global_blocks=2, vocab_blocks=1, k_r=10, no_residual=True)
    model.load_state_dict(torch.load(OUT/"dis_chain_L1.pt", map_location="cpu", weights_only=True))
    model.eval(); spy.__globals__["CAP"].clear()
    with torch.no_grad():
        for s in data.test[:256]:
            seqs, lens, _ = pad_batch([s]); a = build_adjacency(seqs, lens)
            model(seqs, lens, a)
    margins, widths = [], []
    for z in spy.__globals__["CAP"]:
        zz = z.reshape(-1, z.shape[-1])
        for row in zz[::max(1, zz.shape[0]//200)]:
            top = row.topk(min(2, row.numel())).values
            margins.append(float(top[0]-top[-1]))
            w = _real_entmax(row, 1.5, -1); nz = row[w > 0]
            if nz.numel(): widths.append(float(top[0])-float(nz.min()))
    m0, tw = float(np.mean(margins)), float(np.mean(widths))
    lam_eff = min(max(lam, 1e-4), 0.999)
    raw_L = 1 + np.log(m0/tw)/np.log(1/(1-lam_eff))
    pred = int(np.clip(round(raw_L), 1, 4))
    log(f"margin={m0:.4f} tau_width={tw:.4f} raw L*={raw_L:.2f} -> PREDICTED L*={pred}")

    # unblind ONLY after prediction is logged
    for L in (1, 2, 3):
        train(["--data", "data/longchain.pkl", "--model", "mplignn",
               "--lr", "2e-3", "--epochs", "120", "--patience", "15",
               "--max-batches", "0", "--local-layers", L, "--seed", "7",
               "--out", f"dis_chain_full_L{L}.json"], f"dis_chain_full_L{L}.json")
    emp = {}
    for L in (1, 2, 3):
        f = OUT/f"dis_chain_full_L{L}.json"
        if f.exists():
            d = json.load(open(f))["test"]
            emp[L] = {"HR@20": round(100*d["HR@20"], 2),
                      "MRR@20": round(100*d["MRR@20"], 2)}
    best = max(emp, key=lambda L: emp[L]["HR@20"]) if emp else None
    log(f"empirical: {emp} -> best depth {best}")

    res = {"registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "corpus": {"n_items": N_ITEMS, "n_groups": N_GROUPS,
                      "n_train": N_TRAIN, "length": LEN, "seed": SEED},
           "lambda": round(lam, 4), "margin0": round(m0, 4),
           "tau_width": round(tw, 4), "raw_L*": round(float(raw_L), 2),
           "predicted_L*": pred,
           "empirical": emp, "empirical_best_L": best,
           "verdict": ("CONFIRMED" if best == pred else
                       "REFUTED" if best is not None else "PENDING"),
           "wall_minutes": round((time.time()-t_start)/60, 1)}
    (OUT/"discriminating_test.json").write_text(json.dumps(res, indent=2))
    log(f"VERDICT: {res['verdict']} (predicted {pred}, empirical {best})")

if __name__ == "__main__":
    main()
