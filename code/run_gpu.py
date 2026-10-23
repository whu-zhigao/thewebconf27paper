#!/usr/bin/env python3
"""Unified GPU-node runner for the full-epoch (budget-free) protocol.

Plans
-----
  yoo  (default)  The chair-approved GPU scope: the six full-epoch runs
                  on Yoochoose 1/4 that are prohibitively slow on CPU.
  full3           All three full-protocol datasets (Diginetica, Gowalla,
                  Yoo-1/4). Resumable: any full_*.json that already
                  exists (e.g., produced on the workstation) is skipped,
                  so this plan can also finish a partial local matrix.

Protocol (identical to the local full-epoch runs, so results are directly
comparable): no batch cap, 30 epochs, patience 5, batch 128; baselines at
their published size/lr (dim=100, lr=1e-3); MPLI-GNN at its design config
(d=64, lr=1e-3). cudnn is pinned deterministic.

Usage
-----
    python3 run_gpu.py --check                    # device + data sanity
    nohup python3 run_gpu.py > gpu.log 2>&1 &     # default plan (yoo)
    nohup python3 run_gpu.py --plan full3 > gpu.log 2>&1 &
    tail -f gpu.log                               # monitor

After the runs this script (re)packs every full_* result into
gpu_results.tar.gz -- copy that ONE file back to the workstation and run:
    python3 scripts/ingest_gpu_results.py --tar gpu_results.tar.gz
"""
import argparse
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch

P = Path(__file__).parent
OUT = P / "results" / "real"
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if getattr(torch.backends, "mps", None)
       and torch.backends.mps.is_available() else "cpu")

RUNS = [  # (out-stem, extra args) -- cheap first so datasets land early
    ("full_itemknn_{d}",      ["--model", "itemknn"]),
    ("full_narm_{d}",         ["--model", "narm", "--dim", "100", "--lr", "1e-3"]),
    ("full_srgnn_{d}",        ["--model", "srgnn", "--dim", "100", "--lr", "1e-3"]),
    ("full_gcegnn_{d}",       ["--model", "gcegnn", "--dim", "100", "--lr", "1e-3"]),
    ("full_mplignn_L1_{d}",   ["--model", "mplignn", "--ablation", "no_residual",
                               "--lr", "1e-3"]),
    ("full_mplignn_{d}",      ["--model", "mplignn", "--lr", "1e-3"]),
]
BASE = ["--epochs", "30", "--patience", "5", "--max-batches", "0"]
PLANS = {
    "yoo":   {"yoochoose_quarter": "data/yoochoose_quarter.pkl"},
    "full3": {"diginetica": "data/diginetica.pkl",
              "gowalla": "data/gowalla.pkl",
              "yoochoose_quarter": "data/yoochoose_quarter.pkl"},
}


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(P / "gpu_run.log", "a") as f:
        f.write(line + "\n")


def check():
    log(f"device={DEV}")
    if DEV == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} "
            f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    missing = [d for d, p in PLANS["full3"].items() if not (P / p).exists()]
    log(f"datasets ok: {not missing}" + (f" missing={missing}" if missing else ""))
    return DEV != "cpu" and not missing


def pack():
    got = sorted(OUT.glob("full_*.json")) + sorted(OUT.glob("full_*.pt"))
    tar = P / "gpu_results.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in got:
            tf.add(f, arcname=f"results/real/{f.name}")
    log(f"packed {len(got)} files -> {tar.name} ({tar.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS), default="yoo")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not check():
        if args.check:
            sys.exit(1)
        log("WARNING: no GPU or missing data -- aborting (use --check to diagnose)")
        sys.exit(2)
    if args.check:
        log("check OK")
        return
    if DEV == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log(f"GPU RUNNER plan={args.plan} device={DEV}")
    for ds, pkl in PLANS[args.plan].items():
        for stem, extra in RUNS:
            out = OUT / stem.format(d=ds)
            if out.exists():
                log(f"skip {out.name}")
                continue
            t0 = time.time()
            cmd = [sys.executable, "scripts/train.py", "--data", pkl,
                   *extra, *BASE, "--device", DEV, "--out", out]
            log(f"RUN {out.name}")
            r = subprocess.run(cmd, cwd=P, capture_output=True, text=True,
                               errors="replace")
            log(f"  rc={r.returncode} ok={out.exists()} "
                f"{(time.time()-t0)/60:.1f}min")
            if not out.exists():
                (OUT / f"fail_{out.stem}.txt").write_text(
                    (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
                log("  ERR tail: " + (r.stderr or "")[-400:].replace("\n", " "))
    pack()
    log("GPU RUNS COMPLETE -- copy gpu_results.tar.gz back to the workstation")


if __name__ == "__main__":
    main()
