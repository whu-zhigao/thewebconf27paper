#!/usr/bin/env python3
"""Full-epoch (budget-free) protocol runner -- chair remediation W1.

Device policy: cuda -> mps -> cpu (auto). On this workstation the current
torch build (2.0.1) exposes neither, so runs execute on CPU; the protocol
itself removes the 300-batch cap, which is the sampling constraint the
review objected to. When a GPU torch is installed, the same command uses
it with no code change.

Protocol: no batch cap (max-batches=0), 30 epochs, patience 5.
Baselines use their published size/lr (dim=100, lr=1e-3); MPLI-GNN uses
its design config (d=64). Outputs full_<model>_<ds>.json (resumable).
"""
import subprocess, sys, time
from pathlib import Path

import torch

P = Path(__file__).parent
OUT = P/"results"/"real"
LOG = open(P/"fullbudget.log", "a", buffering=1)
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
       else "cpu")

def log(m): LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n"); print(m, flush=True)

def device_args():
    return ["--device", DEV] if DEV != "cpu" else []

# Yoo-1/4 full-epoch deferred to a GPU environment (chair-approved scope cut;
# ~25 CPU-hours per MPLI run). Set GPU_FULL=1 to re-include it when ready.
import os
DS = {"diginetica": "data/diginetica.pkl", "gowalla": "data/gowalla.pkl"}
if os.environ.get("GPU_FULL") == "1":
    DS["yoochoose_quarter"] = "data/yoochoose_quarter.pkl"
RUNS = [  # (out-stem, extra args) -- cheap first so each dataset lands early
    ("full_itemknn_{d}", ["--model", "itemknn"]),
    ("full_narm_{d}",    ["--model", "narm", "--dim", "100", "--lr", "1e-3"]),
    ("full_srgnn_{d}",   ["--model", "srgnn", "--dim", "100", "--lr", "1e-3"]),
    ("full_gcegnn_{d}",  ["--model", "gcegnn", "--dim", "100", "--lr", "1e-3"]),
    ("full_mplignn_L1_{d}", ["--model", "mplignn", "--ablation", "no_residual", "--lr", "1e-3"]),
    ("full_mplignn_{d}",    ["--model", "mplignn", "--lr", "1e-3"]),
    # Denoise-off cells: close the full-epoch 2x2 (residual x denoising).
    # Diginetica + Gowalla are released records (GPU, 2026-10-17, seed 42;
    # also runnable via cloud_nod_bundle/run_nod_cells.py). Yoochoose-1/4
    # pair pending (cloud_round8_bundle/run_round8.py yoo-nod). The loop
    # skips existing outputs.
    ("full_mplignn_L1nod_{d}.json", ["--model", "mplignn", "--ablation", "no_residual,no_denoise", "--lr", "1e-3"]),
    ("full_mplignn_nod_{d}.json",    ["--model", "mplignn", "--ablation", "no_denoise", "--lr", "1e-3"]),
]
BASE = ["--epochs", "30", "--patience", "5", "--max-batches", "0"]

def main():
    log(f"FULL-BUDGET RUNNER device={DEV}")
    for ds, pkl in DS.items():
        for stem, extra in RUNS:
            out = OUT/stem.format(d=ds)
            if out.exists():
                log(f"skip {out.name}"); continue
            t0 = time.time()
            args = ["scripts/train.py", "--data", pkl, *extra, *BASE,
                    *device_args(), "--out", out]
            log(f"RUN {ds} {' '.join(map(str, extra))}")
            r = subprocess.run([sys.executable]+[str(a) for a in args], cwd=P,
                               capture_output=True, text=True, errors="replace")
            ok = out.exists()
            log(f"  -> rc={r.returncode} ok={ok} {(time.time()-t0)/60:.1f}min")
            if not ok:
                (OUT/f"fail_{out.stem}.txt").write_text(
                    (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
                log("  ERR tail: " + (r.stderr or "")[-400:].replace("\n", " "))
    log("FULL-BUDGET COMPLETE")

if __name__ == "__main__":
    main()
