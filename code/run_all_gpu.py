#!/usr/bin/env python3
"""Master GPU orchestrator: run EVERY experiment in the paper on the cloud.

All phases are idempotent and resumable -- every unit writes exactly one
JSON and is skipped if that JSON already exists. Restart freely.

Phases
------
  0  selftest            device + data + numerics sanity (must pass)
  1  budget-matrix       14 models x 6 datasets, budget protocol
  2  five-seed           4 models x 6 datasets x seeds {1..5}
  3  ablation            8 variants + joint, Diginetica (budget)
  4  full-epoch          6 models x 3 datasets, no batch cap, 30 epochs
  5  sensitivity         depth/heads/dim/alpha sweep on Diginetica
  6  noise               5 models x {0..50}% corruption, Diginetica
  7  stratify            shift-stratified eval (MPLI + baselines)
  8  efficiency          GPU efficiency benchmark (Diginetica)
  9  audits              depth-gate audit, theory, discriminating, case
                         study, diagnostics, calibration
  10 pilot               P1/P2/P3 pilot studies on all 6 datasets
  11 pack                everything into gpu_results.tar.gz

Usage
-----
    python3 run_all_gpu.py --check
    python3 run_all_gpu.py                    # all phases in order
    python3 run_all_gpu.py --phases 4,6       # subset, e.g. resume
    nohup python3 run_all_gpu.py > gpu_all.log 2>&1 &
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
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DS6 = ["diginetica", "gowalla", "yoochoose", "yoochoose_quarter",
       "nowplaying", "retailrocket"]
DS3 = ["diginetica", "gowalla", "yoochoose_quarter"]
MODELS14 = ["itemknn", "fpmc", "gru4rec", "narm", "stamp", "srgnn",
            "lessr", "dhcn", "gcegnn", "mtd", "dsan", "msgifsr", "msgat",
            "mplignn"]
SEED_MODELS = ["mplignn_L1", "mplignn", "srgnn", "narm"]
ABLATIONS = ["no_residual", "no_global", "no_anchor", "softmax_attn",
             "no_denoise", "avg_intent", "no_residual,no_denoise"]
BUDGET = ["--lr", "5e-4", "--epochs", "12", "--patience", "4",
          "--max-batches", "300"]
FULLEP = ["--epochs", "30", "--patience", "5", "--max-batches", "0"]
LOG = open(P / "gpu_all.log", "a", buffering=1)


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    LOG.write(line + "\n")
    print(line, flush=True)


def run(cmd, cwd=P):
    import os
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")   # avoid OMP SHM issues in containers
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                       text=True, errors="replace", env=env)
    dt = (time.time() - t0) / 60
    return r, dt


def train_once(out_name, data, extra, tag):
    out = OUT / f"{out_name}.json"
    if out.exists():
        log(f"  skip {out_name}")
        return True
    r, dt = run([sys.executable, "scripts/train.py", "--data", f"data/{data}",
                 *extra, "--device", DEV, "--out", out])
    ok = out.exists()
    log(f"  {'OK ' if ok else 'FAIL'} {out_name} ({dt:.1f}min)")
    if not ok:
        (OUT / f"fail_{out_name}.txt").write_text(
            (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
    return ok


def phase0():
    log("P0 selftest")
    r, _ = run([sys.executable, "scripts/gpu_selftest.py"])
    log(r.stdout[-2000:] if r.returncode == 0 else
        f"SELFTEST FAILED\n{r.stdout[-1000:]}\n{r.stderr[-1500:]}")
    return r.returncode == 0


def phase1():
    log("P1 budget matrix (14 x 6)")
    for ds in DS6:
        for m in MODELS14:
            if m == "itemknn":
                extra = ["--model", "itemknn"]
            elif m == "mplignn":
                extra = ["--model", "mplignn"]
            else:
                extra = ["--model", m]
            train_once(f"{m}_{ds}", ds, extra, "p1")


def phase2():
    log("P2 five-seed (4 models x 6 ds x 5 seeds)")
    for m in SEED_MODELS:
        for ds in DS6:
            for seed in range(1, 6):
                extra = (["--model", "mplignn", "--ablation", "no_residual"]
                         if m == "mplignn_L1" else ["--model", m])
                train_once(f"seed_{m}_{ds}_s{seed}", ds,
                           [*extra, "--seed", str(seed)], "p2")


def phase3():
    log("P3 ablation (Diginetica)")
    for abl in ABLATIONS:
        train_once(f"abl_{abl.replace(',', '_')}_diginetica", "diginetica",
                   ["--model", "mplignn", "--ablation", abl], "p3")


def phase4():
    log("P4 full-epoch (6 x 3)")
    full_runs = [
        ("full_itemknn_{d}", ["--model", "itemknn"]),
        ("full_narm_{d}", ["--model", "narm", "--dim", "100", "--lr", "1e-3"]),
        ("full_srgnn_{d}", ["--model", "srgnn", "--dim", "100", "--lr", "1e-3"]),
        ("full_gcegnn_{d}", ["--model", "gcegnn", "--dim", "100", "--lr", "1e-3"]),
        ("full_mplignn_L1_{d}", ["--model", "mplignn", "--ablation",
                                 "no_residual", "--lr", "1e-3"]),
        ("full_mplignn_{d}", ["--model", "mplignn", "--lr", "1e-3"]),
    ]
    for ds in DS3:
        for stem, extra in full_runs:
            train_once(stem.format(d=ds), ds, [*extra, *FULLEP], "p4")


def phase5():
    log("P5 sensitivity (Diginetica)")
    out = OUT / "sensitivity_diginetica.json"
    if out.exists():
        log("  skip sensitivity")
        return
    r, dt = run([sys.executable, "scripts/sensitivity.py",
                 "--data", "data/diginetica.pkl", "--epochs", "12",
                 "--device", DEV, "--out", out])
    log(f"  {'OK ' if out.exists() else 'FAIL'} sensitivity ({dt:.1f}min)")


def phase6():
    log("P6 noise (Diginetica)")
    for m in ["srgnn", "gcegnn", "dsan", "msgat", "mplignn"]:
        out = OUT / f"noise_{m}_diginetica.json"
        if out.exists():
            log(f"  skip noise_{m}")
            continue
        ck = OUT / (f"mplignn_diginetica.pt" if m == "mplignn"
                    else f"{m}_diginetica.pt")
        if not ck.exists():
            log(f"  missing ckpt {ck.name}")
            continue
        r, dt = run([sys.executable, "scripts/noise_injection.py",
                     "--data", "data/diginetica.pkl", "--ckpt", ck,
                     "--model", m, "--out", out])
        log(f"  {'OK ' if out.exists() else 'FAIL'} noise_{m} ({dt:.1f}min)")


def phase7():
    log("P7 stratify")
    out = OUT / "stratified_diginetica.json"
    if not out.exists():
        r, dt = run([sys.executable, "scripts/stratify_many.py"])
        log(f"  stratify_many {'OK' if out.exists() else 'FAIL'} ({dt:.1f}min)")
    else:
        log("  skip stratify_many")
    out2 = OUT / "stratify_diginetica.json"
    if not out2.exists():
        r, dt = run([sys.executable, "scripts/stratify_shift.py",
                     "--data", "data/diginetica.pkl", "--ckpt",
                     OUT / "mplignn_diginetica.pt", "--out", out2])
        log(f"  stratify_shift {'OK' if out2.exists() else 'FAIL'}")


def phase8():
    log("P8 efficiency (GPU)")
    for ds in DS3:
        out = OUT / f"efficiency_{ds}_gpu.json"
        if out.exists():
            log(f"  skip efficiency_{ds}")
            continue
        r, dt = run([sys.executable, "scripts/efficiency.py",
                     "--data", f"data/{ds}.pkl", "--out", out])
        log(f"  {'OK ' if out.exists() else 'FAIL'} efficiency_{ds} ({dt:.1f}min)")


def phase9():
    log("P9 audits")
    jobs = [
        ("depth_gate_audit", ["scripts/depth_gate_audit.py"],
         OUT / "audit_depthgate.json"),
        ("theory_validation", ["scripts/theory_validation.py"],
         OUT / "theory_validation.json"),
        ("discriminating", ["scripts/discriminating_test.py"],
         OUT / "dis_chain_L1.json"),
        ("case_study", ["scripts/case_study.py"],
         OUT / "case_study_diginetica.json"),
        ("diagnostics", ["scripts/diagnostics_readout.py", "--data",
                         "data/diginetica.pkl", "--ckpt",
                         OUT / "mplignn_diginetica.pt", "--out",
                         OUT / "diagnostics_diginetica.json"],
         OUT / "diagnostics_diginetica.json"),
        ("calibration", ["scripts/calibration_transfer.py"],
         OUT / "calibration_transfer.json"),
    ]
    for name, cmd, marker in jobs:
        if marker.exists():
            log(f"  skip {name}")
            continue
        r, dt = run([sys.executable, *cmd])
        log(f"  {'OK ' if marker.exists() else 'FAIL'} {name} ({dt:.1f}min)")


def phase10():
    log("P10 pilot studies")
    for ds in DS6:
        out = OUT / f"pilot_{ds}.json"
        if out.exists():
            log(f"  skip pilot_{ds}")
            continue
        r, dt = run([sys.executable, "scripts/pilot_study.py",
                     "--data", f"data/{ds}.pkl", "--out", out])
        log(f"  {'OK ' if out.exists() else 'FAIL'} pilot_{ds} ({dt:.1f}min)")


def phase11():
    log("P11 pack")
    pats = ["full_*.json", "full_*.pt", "seed_*.json", "sens_*.json",
            "noise_*.json", "stratif*.json", "efficiency_*_gpu.json",
            "audit*.json", "theory*.json", "dis_*.json", "case_*.json",
            "diagnostics*.json", "calibration*.json", "pilot_*.json"]
    files = []
    for pat in pats:
        files += sorted(OUT.glob(pat))
    files = sorted(set(files))
    tar = P / "gpu_results.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=f"results/real/{f.name}")
    log(f"packed {len(files)} files -> {tar.name} "
        f"({tar.stat().st_size/1e6:.1f} MB)")
    log("ALL PHASES COMPLETE -- copy gpu_results.tar.gz back")


PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3, 4: phase4,
          5: phase5, 6: phase6, 7: phase7, 8: phase8, 9: phase9,
          10: phase10, 11: phase11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--phases", default=",".join(str(i) for i in PHASES))
    args = ap.parse_args()
    log(f"MASTER ORCHESTRATOR device={DEV} phases={args.phases}")
    if args.check:
        phase0()
        return
    want = [int(x) for x in args.phases.split(",") if x != ""]
    for i in want:
        PHASES[i]()
    log("orchestrator finished requested phases")


if __name__ == "__main__":
    main()
