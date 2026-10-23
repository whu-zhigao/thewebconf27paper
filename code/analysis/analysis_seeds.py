#!/usr/bin/env python3
"""Seed-level statistics for the revised paper.

Reads the released per-run JSON records in results/real and emits genuine
mean/std plus paired significance tests. No scipy dependency: the two-sided
paired t-test uses a self-contained incomplete-beta tail, and the Wilcoxon
signed-rank test is enumerated exactly (n is tiny and ties are absent).
"""
import glob
import json
import math
import itertools
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "real"


# ----------------------------- statistics ---------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((qap + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p-value of Student's t with df degrees of freedom."""
    return betai(0.5 * df, 0.5, df / (df + t * t))


def paired_t(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return (float("inf"), 0.0) if d.mean() != 0 else (0.0, 1.0)
    t = d.mean() / (sd / math.sqrt(n))
    return t, t_sf_two_sided(t, n - 1)


def wilcoxon_exact(x, y):
    """Exact two-sided Wilcoxon signed-rank p (ties absent by construction)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    w_obs = r[d > 0].sum()
    total, extreme = 0, 0
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(ri for ri, s in zip(r, signs) if s)
        total += 1
        if w <= w_obs or w >= n * (n + 1) / 2 - w_obs:
            extreme += 1
    return extreme / total


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ------------------------------- loading ----------------------------------
def load(prefix_glob):
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(str(RES / prefix_glob))):
        d = json.load(open(f))
        key = (d["model"], d["dataset"], tuple(sorted(d.get("ablation", {}).items())))
        out[key].append(d)
    return out


def summarize(runs):
    print(f"{'model':9s} {'dataset':18s} {'config':26s} {'n':>2s}  "
          f"{'HR@20':>15s}  {'MRR@20':>15s}  best-epochs")
    table = {}
    for k in sorted(runs, key=str):
        rs = sorted(runs[k], key=lambda d: d["seed"])
        hr = np.array([d["test"]["HR@20"] * 100 for d in rs])
        mrr = np.array([d["test"]["MRR@20"] * 100 for d in rs])
        cfg = ",".join(f"{a}={b}" for a, b in k[2]) or "full(L=2)"
        table[(k[0], k[1], cfg)] = (hr, mrr, rs)
        print(f"{k[0]:9s} {k[1]:18s} {cfg:26s} {len(rs):2d}  "
              f"{hr.mean():6.2f}±{hr.std(ddof=1):4.2f}  "
              f"{mrr.mean():6.2f}±{mrr.std(ddof=1):4.2f}  "
              f"{[d.get('best_epoch') for d in rs]}")
    return table


if __name__ == "__main__":
    runs = load("seed_*.json")
    print("=== released multi-seed records ===")
    table = summarize(runs)

    comparisons = [
        ("MPLI-L1 vs MPLI-L2", ("mplignn", "no_residual=true"), ("mplignn", "full(L=2)")),
        ("MPLI-L1 vs NARM", ("mplignn", "no_residual=true"), ("narm", "full(L=2)")),
        ("MPLI-L1 vs SR-GNN", ("mplignn", "no_residual=true"), ("srgnn", "full(L=2)")),
        ("MPLI-L2 vs NARM", ("mplignn", "full(L=2)"), ("narm", "full(L=2)")),
    ]
    for j, metric in enumerate(["HR@20", "MRR@20"]):
        print(f"\n=== paired comparisons, {metric} (Holm across all rows) ===")
        raw, lines = [], []
        for label, ka, kb in comparisons:
            for ds in sorted({k[1] for k in table}):
                a = table.get((ka[0], ds, ka[1]))
                b = table.get((kb[0], ds, kb[1]))
                if a is None or b is None:
                    continue
                x, y = a[j], b[j]
                t, p = paired_t(x, y)
                wp = wilcoxon_exact(x, y)
                raw.append(p)
                lines.append((label, ds, x.mean(), y.mean(), x.mean() - y.mean(), t, p, wp, len(x)))
        for (label, ds, ma, mb, diff, t, p, wp, n), pa in zip(lines, holm(raw)):
            star = "*" if pa < 0.05 else " "
            print(f"{star} {label:20s} {ds:18s} n={n} {ma:6.2f} vs {mb:6.2f} "
                  f"diff={diff:+6.2f} t={t:6.2f} p={p:.4f} p_holm={pa:.4f} p_wilcox={wp:.4f}")
