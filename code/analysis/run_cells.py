#!/usr/bin/env python3
"""Resumable driver for the revision's controlled depth x denoising design.

The original ablation conflated two factors: `--ablation no_residual` keeps
TWO local layers but removes the residual, whereas `--local-layers 1` uses a
single layer WITH a residual. Only the former was ever run, so the 2x2 design
the paper claims to report was never measured. This driver fills the missing
cells under the identical budget (lr 5e-4, 12 epochs, patience 4, <=300
batches/epoch) and is resumable: existing JSONs are skipped.

Cells
  D0R0 : L=2, no residual      -> --ablation no_residual
  D0R1 : L=2, residual         -> (default)
  D1R1 : L=1, residual         -> --local-layers 1
  denoise depends on the second flag (no_denoise) for each.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "results" / "real"
PY = sys.executable
BUDGET = ["--lr", "5e-4", "--epochs", "12", "--patience", "4", "--max-batches", "300"]
CLIP_BUDGET = ["--lr", "5e-5", "--epochs", "12", "--patience", "4", "--max-batches", "300"]

# (tag, extra model flags)
CELLS = {
    "D0R0": ["--ablation", "no_residual"],
    "D0R0nod": ["--ablation", "no_residual,no_denoise"],
    "D0R1": [],
    "D0R1nod": ["--ablation", "no_denoise"],
    "D1R1": ["--local-layers", "1"],
    "D1R1nod": ["--local-layers", "1", "--ablation", "no_denoise"],
    # learning-rate sensitivity: does the depth-2 penalty survive a 10x
    # smaller step size, i.e. is it an optimization artifact?
    "LRlowD0R1": ["--lr", "5e-5"],
    "LRlowD1R1": ["--lr", "5e-5", "--local-layers", "1"],
    # depth-1 WITHOUT residual: completes the 2x2 (depth x residual) design
    "D1R0": ["--local-layers", "1", "--ablation", "no_residual"],
    "D1R0nod": ["--local-layers", "1", "--ablation", "no_residual,no_denoise"],
}

DATASETS = ["diginetica", "gowalla", "yoochoose_quarter", "nowplaying",
            "retailrocket", "yoochoose"]


def run_one(dataset, cell, seed, threads):
    out = OUT / f"{cell}_{dataset}_s{seed}.json"
    if out.exists() and out.stat().st_size > 0:
        try:
            d = json.load(open(out))
            return True, f"skip {out.name} (HR@20={d['test']['HR@20']*100:.2f})"
        except Exception:
            pass
    flags = list(CELLS[cell])
    budget = [b for b in BUDGET if not (b == "--lr" or b == "5e-4")] \
        if "--lr" in flags else BUDGET
    cmd = [PY, str(CODE / "scripts" / "train.py"),
           "--data", str(CODE / "data" / f"{dataset}.pkl"),
           "--model", "mplignn", *flags, *budget,
           "--seed", str(seed), "--out", str(out)]
    env = {"PYTHONPATH": str(CODE), "OMP_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if p.returncode != 0:
        return False, f"FAIL {out.name}: {p.stderr.strip().splitlines()[-1] if p.stderr else '?'}"
    d = json.load(open(out))
    return True, (f"{out.name:34s} HR@20={d['test']['HR@20']*100:6.2f} "
                  f"MRR@20={d['test']['MRR@20']*100:6.2f} "
                  f"best_ep={d.get('best_epoch'):2d} ({dt/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=4, type=int)
    ap.add_argument("--seeds", default="42,7,13,21,34")
    ap.add_argument("--datasets", default="diginetica,gowalla")
    ap.add_argument("--cells", default="D1R1,D1R1nod,D0R1nod,D0R0,D0R0nod,D0R1")
    ap.add_argument("--threads", default=2, type=int)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--cell-datasets", default="",
                    help="semicolon list of cell:dataset pairs to restrict to")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d for d in args.datasets.split(",") if d.strip()]
    cells = [c for c in args.cells.split(",") if c.strip()]

    # priority order: fill L=1 (never measured) first, then the denoise toggles
    jobs = [(d, c, s) for c in cells for d in datasets for s in seeds]
    if args.cell_datasets:
        allow = set()
        for part in args.cell_datasets.split(";"):
            if not part.strip():
                continue
            c, d = part.split(":")
            allow.add((c.strip(), d.strip()))
        jobs = [j for j in jobs if (j[1], j[0]) in allow]
    print(f"plan: {len(jobs)} runs, {args.jobs} workers, threads={args.threads}")
    if args.plan_only:
        for j in jobs:
            print("  ", j)
        return

    # simple worker pool
    from concurrent.futures import ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, d, c, s, args.threads): (d, c, s)
                for (d, c, s) in jobs}
        for f in futs:
            ds, cell, seed = futs[f]
            ok, msg = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
            results.append((ds, cell, seed, ok))
    bad = [r for r in results if not r[3]]
    print(f"\nDONE: ok={len(results)-len(bad)} fail={len(bad)}")
    for r in bad:
        print("  FAILED:", r)


if __name__ == "__main__":
    main()
