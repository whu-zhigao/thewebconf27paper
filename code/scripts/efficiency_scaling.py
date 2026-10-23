#!/usr/bin/env python3
# M7 efficiency scaling: K_r sweep + dead-module cost accounting.
#
# Produces results/systems/efficiency_scaling.json with, per dataset:
#   * for K_r in {5,10,20,40}: median ms/train-step, ms/test-session,
#     and peak RSS;
#   * a cross-variant timing comparison of the full instrument against
#     the no_global+no_anchor variant (cost accounting for the two
#     zero-gradient modules; prediction identity itself is certified by
#     analysis/probe_dead_paths.py, not by this script).
#
# The K_r sweep relies on the k_r-specific relcache naming introduced in
# the revision (scripts/train.py); missing indices are built on first use.
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import (FastRelationIndex, load_sessions, pad_batch,
                          build_adjacency)
from mplignn.model import MPLI_GNN


def peak_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def median_step_ms(model, data, rel, device, n_steps=40, batch_size=128):
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(0)
    times = []
    for _ in range(n_steps):
        ids = torch.from_numpy(rng.choice(len(data.train), batch_size,
                                          replace=False))
        chunk = [data.train[j] for j in ids.tolist()]
        seqs, lens, targets = pad_batch(chunk, device)
        a = build_adjacency(seqs, lens)
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(seqs, lens, a, a_r, r0)
        loss = F.cross_entropy(out["scores"], targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


@torch.no_grad()
def eval_latency(model, data, rel, device, n_sessions=2000, batch_size=128):
    model.eval()
    sessions = data.test[:n_sessions]
    t0 = time.perf_counter()
    for i in range(0, len(sessions), batch_size):
        chunk = sessions[i:i + batch_size]
        ids = torch.arange(i, i + len(chunk))
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        seqs, lens, targets = pad_batch(chunk, device)
        a = build_adjacency(seqs, lens)
        model(seqs, lens, a, a_r, r0)
    dt = time.perf_counter() - t0
    return dt * 1000 / max(len(sessions), 1)


def _train_cache(ds, k_r):
    relcache = Path("data/relcache")
    cand = relcache / f"{ds}_train_k{k_r}.npz"
    if cand.exists():
        return cand
    if k_r == 10:
        legacy = relcache / f"{ds}_train.npz"
        if legacy.exists():
            return legacy
    return cand


def sweep_dataset(ds, device, out):
    data = load_sessions(Path(f"data/{ds}.pkl"))
    relcache = Path("data/relcache"); relcache.mkdir(exist_ok=True)
    rows = {}
    for k_r in (5, 10, 20, 40):
        rel = FastRelationIndex(data.train, data.train, data.n_items, k_r,
                                cache=_train_cache(ds, k_r))
        torch.manual_seed(42); np.random.seed(42)
        model = MPLI_GNN(n_items=data.n_items).to(device)
        step_ms = median_step_ms(model, data, rel, device)
        tc = relcache / f"{ds}_test_k{k_r}.npz"
        if not tc.exists() and k_r == 10:
            legacy = relcache / f"{ds}_test.npz"
            if legacy.exists():
                tc = legacy
        rel_te = FastRelationIndex(data.test, data.train, data.n_items,
                                   k_r, cache=tc)
        ms_sess = eval_latency(model, data, rel_te, device)
        rows[f"k{k_r}"] = {"step_ms": round(step_ms, 2),
                           "ms_per_session": round(ms_sess, 3),
                           "peak_rss_mb": round(peak_rss_mb(), 1)}
        print(ds, f"k={k_r}", rows[f"k{k_r}"], flush=True)

    rel10 = FastRelationIndex(data.train, data.train, data.n_items, 10,
                              cache=_train_cache(ds, 10))
    torch.manual_seed(42); np.random.seed(42)
    full = MPLI_GNN(n_items=data.n_items).to(device)
    torch.manual_seed(42); np.random.seed(42)
    clean = MPLI_GNN(n_items=data.n_items, no_global=True,
                     no_anchor=True).to(device)
    full_ms = median_step_ms(full, data, rel10, device)
    clean_ms = median_step_ms(clean, data, rel10, device)
    rows["dead_modules"] = {
        "full_step_ms": round(full_ms, 2),
        "clean_step_ms": round(clean_ms, 2),
        "overhead_pct": round(100 * (full_ms / clean_ms - 1), 1),
        "note": "cross-variant timing; identity certified by "
                "analysis/probe_dead_paths.py"}
    out[ds] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="diginetica,gowalla")
    ap.add_argument("--out", type=Path,
                    default=Path("results/systems/efficiency_scaling.json"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = {}
    if args.out.exists():
        out = json.loads(args.out.read_text())
    for ds in args.datasets.split(","):
        if ds in out:
            print(f"skip {ds} (exists)")
            continue
        sweep_dataset(ds, args.device, out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
