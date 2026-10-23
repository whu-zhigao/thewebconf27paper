#!/usr/bin/env python3
"""GPU-node runner: full-epoch protocol on Yoo-1/4 ONLY.

Usage on the GPU node:
    cd code && python3 run_gpu_yoochoose.py            # normal run
    python3 run_gpu_yoochoose.py --check               # device + data sanity only

Protocol is IDENTICAL to the CPU full-epoch runs (batch 128, 30 epochs,
patience 5, no batch cap, published baseline configs) so results are
directly comparable; the GPU only accelerates execution. cudnn is pinned
deterministic for reproducibility. After the runs, this script packs
results/real/full_*_yoochoose_quarter*.json (+ .pt) into
gpu_results.tar.gz -- copy that single file back to the workstation.
"""
import argparse, subprocess, sys, tarfile, time
from pathlib import Path
import torch

P = Path(__file__).parent
OUT = P/"results"/"real"
LOG = open(P/"gpu_yoochoose.log", "a", buffering=1)
DEV = "cuda" if torch.cuda.is_available() else (
      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

RUNS = [
    ("full_itemknn_yoochoose_quarter",  ["--model","itemknn"]),
    ("full_narm_yoochoose_quarter",     ["--model","narm","--dim","100","--lr","1e-3"]),
    ("full_srgnn_yoochoose_quarter",    ["--model","srgnn","--dim","100","--lr","1e-3"]),
    ("full_gcegnn_yoochoose_quarter",   ["--model","gcegnn","--dim","100","--lr","1e-3"]),
    ("full_mplignn_L1_yoochoose_quarter", ["--model","mplignn","--ablation","no_residual","--lr","1e-3"]),
    ("full_mplignn_yoochoose_quarter",     ["--model","mplignn","--lr","1e-3"]),
]
BASE = ["--epochs","30","--patience","5","--max-batches","0"]

def log(m): LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n"); print(m, flush=True)

def check():
    log(f"device={DEV}")
    if DEV == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    data = P/"data"/"yoochoose_quarter.pkl"
    log(f"data exists: {data.exists()} ({data.stat().st_size/1e6:.0f} MB)" if data.exists()
        else f"MISSING {data} -- run scripts/prepare_yoochoose.py first")
    return DEV != "cpu" and data.exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not check():
        if args.check: sys.exit(1)
        log("WARNING: continuing on CPU -- Yoo-1/4 full-epoch on CPU takes ~25h/model")
    elif args.check:
        log("check OK"); sys.exit(0)
    if DEV == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    for stem, extra in RUNS:
        out = OUT/f"{stem}.json"
        if out.exists(): log(f"skip {stem}"); continue
        t0 = time.time()
        cmd = [sys.executable, "scripts/train.py", "--data",
               "data/yoochoose_quarter.pkl", *extra, *BASE,
               "--device", DEV, "--out", out]
        log(f"RUN {stem}")
        r = subprocess.run(cmd, cwd=P, capture_output=True, text=True, errors="replace")
        log(f"  rc={r.returncode} ok={out.exists()} {(time.time()-t0)/60:.1f}min")
        if not out.exists():
            (OUT/f"fail_{stem}.txt").write_text((r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
    # pack results for transfer back
    got = sorted(OUT.glob("full_*yoochoose_quarter*.json")) + \
          sorted(OUT.glob("full_*yoochoose_quarter*.pt"))
    with tarfile.open(P/"gpu_results.tar.gz", "w:gz") as tf:
        for f in got: tf.add(f, arcname=f"results/real/{f.name}")
    log(f"packed {len(got)} files -> gpu_results.tar.gz ({(P/'gpu_results.tar.gz').stat().st_size/1e6:.1f} MB)")
    log("GPU RUNS COMPLETE")

if __name__ == "__main__":
    main()
