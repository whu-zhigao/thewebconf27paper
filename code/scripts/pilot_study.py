#!/usr/bin/env python3
"""Preliminary empirical study (Section 5.1, P1-P3).

P1: fraction of multi-intent sessions, measured by clustering item
    co-occurrence embeddings and counting sessions spanning >= 2 clusters.
P2: fraction of shift sessions, measured by cosine drift between the mean
    embedding of the first and last thirds of a session (75th-percentile
    threshold), plus the HR@20 gap of a co-occurrence predictor between
    shift and stable sessions.
P3: fraction of sessions whose ground-truth next item is more similar to
    the local intent (mean of session prefix) than to the global
    preference (mean over the co-occurrence neighborhood of the session).

All thresholds are released with the script; on synthetic data the planted
shift labels are additionally reported as an oracle sanity check.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import load_sessions


def cooccurrence_embeddings(train, n_items, dim=32, seed=0):
    """Random-projection co-occurrence item embeddings (sparse, memory safe).

    Equivalent to counts @ proj with counts the (never materialized)
    item-item co-occurrence matrix: each transition (a, b) adds proj[b] to
    emb[a] and proj[a] to emb[b].
    """
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_items, dim)) / dim ** 0.5
    emb = np.zeros((n_items, dim), dtype=np.float32)
    for s in train:
        for a_i, b_i in zip(s[:-1], s[1:]):
            emb[a_i] += proj[b_i]
            emb[b_i] += proj[a_i]
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    return emb


def kmeans_labels(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for c in range(k):
            if (labels == c).any():
                centers[c] = x[labels == c].mean(0)
    return labels


def build_sparse_transitions(train, n_items):
    """Sparse item->neighbor transition store (memory safe)."""
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(float))
    for s in train:
        for a_i, b_i in zip(s[:-1], s[1:]):
            acc[a_i][b_i] += 1.0
    return {i: (np.fromiter(d.keys(), dtype=np.int64),
                np.fromiter(d.values(), dtype=np.float32))
            for i, d in acc.items()}


def transition_row(transition, n_items, prefix):
    row = np.zeros(n_items, dtype=np.float32)
    for it in prefix:
        if it in transition:
            nb, cnt = transition[it]
            np.add.at(row, nb, cnt)
    return row


def local_hr20(sessions, emb, transition, n_items):
    """Co-occurrence predictor: rank items by transition mass from prefix."""
    hits = 0
    for s in sessions:
        scores = transition_row(transition, n_items, s[:-1])
        top = np.argpartition(-scores, 20)[:20]
        if s[-1] in top:
            hits += 1
    return hits / max(len(sessions), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-clusters", default=20, type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load_sessions(args.data)
    n_items = data.n_items
    emb = cooccurrence_embeddings(data.train, n_items)
    labels = kmeans_labels(emb, args.n_clusters)

    # ---- P1: multi-intent sessions -------------------------------------
    multi = sum(1 for s in data.test if len({labels[i] for i in s}) >= 2)
    p1 = multi / max(len(data.test), 1)

    # ---- P2: shift sessions + HR gap ------------------------------------
    drifts, shift_flags = [], []
    for s in data.test:
        t3 = max(1, len(s) // 3)
        head = emb[s[:t3]].mean(0)
        tail = emb[s[-t3:]].mean(0)
        drift = float(head @ tail)
        drifts.append(drift)
    drifts = np.array(drifts)
    thr = np.percentile(drifts, 25)  # low cosine similarity => shift
    shift_flags = drifts < thr

    transition = build_sparse_transitions(data.train, n_items)
    shift_sessions = [s for s, f in zip(data.test, shift_flags) if f]
    stable_sessions = [s for s, f in zip(data.test, shift_flags) if not f]
    hr_shift = local_hr20(shift_sessions, emb, transition, n_items)
    hr_stable = local_hr20(stable_sessions, emb, transition, n_items)
    p2_frac = float(shift_flags.mean())
    p2_gap = float(hr_stable - hr_shift)

    # ---- P3: local vs global next-item similarity ------------------------
    local_wins = 0
    for s in data.test:
        prefix, nxt = s[:-1], s[-1]
        local_intent = emb[prefix].mean(0)
        neigh = transition_row(transition, n_items, prefix)
        neigh_ids = np.nonzero(neigh)[0]
        global_pref = emb[neigh_ids].mean(0) if len(neigh_ids) else np.zeros_like(local_intent)
        if local_intent @ emb[nxt] >= global_pref @ emb[nxt]:
            local_wins += 1
    p3 = local_wins / max(len(data.test), 1)

    out = {
        "dataset": str(args.data),
        "n_test": len(data.test),
        "P1_multi_intent_fraction": round(p1, 4),
        "P2_shift_fraction": round(p2_frac, 4),
        "P2_hr_stable": round(hr_stable, 4),
        "P2_hr_shift": round(hr_shift, 4),
        "P2_hr_gap": round(p2_gap, 4),
        "P3_local_dominance_fraction": round(p3, 4),
    }
    if data.shift_flags is not None and len(data.shift_flags) == len(data.test):
        oracle = data.shift_flags
        keep = np.array([len(s) >= 2 for s in data.test])
        agree = float((oracle[keep] == shift_flags[keep]).mean())
        out["oracle_agreement_with_detector"] = round(agree, 4)
        out["oracle_shift_fraction"] = round(float(oracle.mean()), 4)

    print(json.dumps(out, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
