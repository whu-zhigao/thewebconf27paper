#!/usr/bin/env python3
"""Cross-architecture depth-gating audit (BPA field-level evidence).

Question: is "depth gates the benefit of global evidence" a property of
MPLI-GNN only, or of SBR architectures in general?
Design: SR-GNN (intra-session only) depth in {1,2,3} as the depth-only
control; GCE-GNN (local+global) depth in {1,2,3} x {global on/off}, so
Delta_global(L) = HR@20(global) - HR@20(no global) isolates the benefit
of global evidence at each depth. Datasets: Diginetica, Gowalla, Yoo-1/4.
Protocol: identical budgeted settings as Table 4 (lr 5e-4, dim 64,
12 epochs, patience 4, 300 batches). Resumable; writes
audit_summary.json with per-architecture gating thresholds.
"""
import json, subprocess, sys, time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P/"results"/"real"
LOG = open(P/"phase_d_audit.log", "a", buffering=1)
DS = {"diginetica": "data/diginetica.pkl", "gowalla": "data/gowalla.pkl",
      "yoochoose_quarter": "data/yoochoose_quarter.pkl"}
BASE = ["--lr", "5e-4", "--epochs", "12", "--patience", "4",
        "--max-batches", "300"]
import torch as _t
_DEV = "cuda" if _t.cuda.is_available() else "cpu"
if _DEV == "cuda":
    BASE += ["--device", "cuda"]

def log(m): LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n"); print(m, flush=True)

def run(model, ds, out_name, extra):
    out = OUT/out_name
    if out.exists(): log(f"skip {out_name}"); return
    t0 = time.time()
    args = ["scripts/train.py", "--data", DS[ds], "--model", model, *extra,
            *BASE, "--out", out]
    log(f"RUN {out_name}")
    r = subprocess.run([sys.executable]+[str(a) for a in args], cwd=P,
                       capture_output=True, text=True, errors="replace")
    log(f"  -> rc={r.returncode} ok={out.exists()} {(time.time()-t0)/60:.1f}min")
    if not out.exists():
        (OUT/f"fail_{out.stem}.txt").write_text((r.stdout or "")[-3000:] + (r.stderr or "")[-3000:])

def main():
    log("DEPTH-GATE AUDIT start")
    for ds in DS:
        for L in (1, 2, 3):
            run("srgnn", ds, f"audit_srgnn_{ds}_L{L}.json",
                ["--depth", L])
            for glob, tag in ((True, "glob"), (False, "noglob")):
                extra = ["--depth", L] + ([] if glob else ["--no-global"])
                run("gcegnn", ds, f"audit_gcegnn_{ds}_L{L}_{tag}.json", extra)
    # summary: Delta_global(L) per ds + SR-GNN depth trend
    S = {"per_dataset": {}}
    for ds in DS:
        row = {"srgnn": {}, "gcegnn_delta_global": {}}
        for L in (1, 2, 3):
            f = OUT/f"audit_srgnn_{ds}_L{L}.json"
            if f.exists():
                row["srgnn"][L] = round(100*json.load(open(f))["test"]["HR@20"], 2)
            g = OUT/f"audit_gcegnn_{ds}_L{L}_glob.json"
            ng = OUT/f"audit_gcegnn_{ds}_L{L}_noglob.json"
            if g.exists() and ng.exists():
                row["gcegnn_delta_global"][L] = round(
                    100*json.load(open(g))["test"]["HR@20"]
                    - 100*json.load(open(ng))["test"]["HR@20"], 2)
        S["per_dataset"][ds] = row
    (OUT/"audit_summary.json").write_text(json.dumps(S, indent=2))
    log("AUDIT COMPLETE: " + json.dumps(S))

if __name__ == "__main__":
    main()
