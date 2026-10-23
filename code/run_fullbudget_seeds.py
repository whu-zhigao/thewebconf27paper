#!/usr/bin/env python3
"""Multi-seed extension of the full-epoch protocol (chair remediation, round 2).

Purpose: upgrade every full-epoch cell in Table tab:fullepoch from
single-seed (42) to mean+-std over seeds {42, 1, 2}, resolving the
"one seed per cell" caveat. Mirrors run_fullbudget.py exactly:
same protocol (no batch cap, 30 epochs, patience 5), same baselines at
published size/lr, same instrument configs at d=64 -- but adds --seed.

Run-stem convention (backwards compatible):
    full_<model>_<ds>_s<seed>.json        (seed != 42)
    full_<model>_<ds>.json                (seed 42, existing runs)

Device policy: cuda -> mps -> cpu (auto). Yoo-1/4 is GPU-gated behind
GPU_FULL=1 exactly as in run_fullbudget.py (~25 CPU-hours per MPLI run;
Diginetica/Gowalla are ~5.6 h each on the reference workstation).

Usage:
    python3 run_fullbudget_seeds.py            # Digi+Gowa, seeds 1,2
    GPU_FULL=1 python3 run_fullbudget_seeds.py # + Yoochoose-1/4
    SEEDS="1 2 7" python3 run_fullbudget_seeds.py
    MODELS=mplignn python3 run_fullbudget_seeds.py   # instrument only

After the runs land, regenerate the table cells with:
    python3 make_fullepoch_seeds_table.py   (see bottom of this file)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

P = Path(__file__).parent
OUT = P / "results" / "real"
LOG = open(P / "fullbudget_seeds.log", "a", buffering=1)
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
       else "cpu")

def log(m):
    LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n")
    print(m, flush=True)

def device_args():
    return ["--device", DEV] if DEV != "cpu" else []

DS = {"diginetica": "data/diginetica.pkl", "gowalla": "data/gowalla.pkl"}
if os.environ.get("GPU_FULL") == "1":
    DS["yoochoose_quarter"] = "data/yoochoose_quarter.pkl"

SEEDS = [int(s) for s in os.environ.get("SEEDS", "1 2").split()]
MODELS = os.environ.get("MODELS", "itemknn,narm,srgnn,gcegnn,mplignn_L1,mplignn").split(",")

# (out-stem, extra args) -- identical to run_fullbudget.py, + --seed added at launch
RUNS = [
    ("full_itemknn_{d}",   ["--model", "itemknn"]),
    ("full_narm_{d}",      ["--model", "narm", "--dim", "100", "--lr", "1e-3"]),
    ("full_srgnn_{d}",     ["--model", "srgnn", "--dim", "100", "--lr", "1e-3"]),
    ("full_gcegnn_{d}",    ["--model", "gcegnn", "--dim", "100", "--lr", "1e-3"]),
    # NOTE on naming: the stem "..._L1" is the historical name of the
    # --ablation no_residual run (two layers, residual removed); see the
    # paper's checkpoint-identification appendix. Do not rename: the JSON
    # carries the authoritative flags.
    ("full_mplignn_L1_{d}", ["--model", "mplignn", "--ablation", "no_residual", "--lr", "1e-3"]),
    ("full_mplignn_{d}",    ["--model", "mplignn", "--lr", "1e-3"]),
]
BASE = ["--epochs", "30", "--patience", "5", "--max-batches", "0"]

def stem_matches(stem, model):
    return stem.split("_")[1] == model or (model == "mplignn_L1" and stem.startswith("full_mplignn_L1")) or \
           (model == "mplignn" and stem.startswith("full_mplignn_") and not stem.startswith("full_mplignn_L1")) or \
           (model == "itemknn" and stem.startswith("full_itemknn")) or \
           (model in ("narm", "srgnn", "gcegnn") and stem.startswith(f"full_{model}"))

def main():
    log(f"FULL-BUDGET SEEDS runner device={DEV} seeds={SEEDS} models={MODELS} datasets={list(DS)}")
    for ds, pkl in DS.items():
        for stem, extra in RUNS:
            if not any(stem_matches(stem, m) for m in MODELS):
                continue
            for seed in SEEDS:
                out = OUT / stem.format(d=ds) if seed == 42 else OUT / (stem.format(d=ds) + f"_s{seed}")
                if out.exists():
                    log(f"skip {out.name}")
                    continue
                t0 = time.time()
                args = ["scripts/train.py", "--data", pkl, *extra, *BASE,
                        "--seed", str(seed), *device_args(), "--out", out]
                log(f"RUN {ds} seed={seed} {' '.join(extra)}")
                r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=P,
                                   capture_output=True, text=True, errors="replace")
                ok = out.exists()
                log(f"  -> rc={r.returncode} ok={ok} {(time.time()-t0)/60:.1f}min")
                if not ok:
                    (OUT / f"fail_{out.stem}.txt").write_text(
                        (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
                    log("  ERR tail: " + (r.stderr or "")[-400:].replace("\n", " "))
    log("FULL-BUDGET SEEDS COMPLETE")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# make_fullepoch_seeds_table.py -- after the runs, print the LaTeX cells.
# Save the block below as its own file, or run:
#   python3 - <<'EOF'  (paste)
#
# import json, glob, statistics as st
# cells = {}
# for stem, label in [("full_itemknn_{d}","Item-KNN"),("full_narm_{d}","NARM"),
#                     ("full_srgnn_{d}","SR-GNN"),("full_gcegnn_{d}","GCE-GNN"),
#                     ("full_mplignn_L1_{d}","no-res"),("full_mplignn_{d}","res")]:
#     for ds in ["diginetica","gowalla","yoochoose_quarter"]:
#         vals = {"hr": [], "mrr": []}
#         for f in [f"results/real/{stem.format(d=ds)}.json"] + \
#                  glob.glob(f"results/real/{stem.format(d=ds)}_s*.json"):
#             try:
#                 t = json.load(open(f))["test"]
#                 vals["hr"].append(100*t["HR@20"]); vals["mrr"].append(100*t["MRR@20"])
#             except Exception:
#                 pass
#         n = len(vals["hr"])
#         if n:
#             cells[(label, ds)] = (st.mean(vals["hr"]), st.stdev(vals["hr"]) if n>1 else 0.0,
#                                   st.mean(vals["mrr"]), st.stdev(vals["mrr"]) if n>1 else 0.0, n)
# for k, v in sorted(cells.items()):
#     print(f"{k[0]:8s} {k[1]:18s} n={v[4]}  "
#           f"{v[0]:.2f}+-{v[1]:.2f} / {v[2]:.2f}+-{v[3]:.2f}")
# EOF
