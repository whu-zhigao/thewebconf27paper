#!/usr/bin/env python3
"""Generate every empirical LaTeX table for the revision from result JSONs.

Cell names encode the factors:
  D0/D1 = local depth 2/1 ; R0/R1 = residual absent/present
  nod   = denoising path OFF (default ON)
`--ablation no_residual` is D0R0 (depth TWO, no residual);
`--local-layers 1` is D1R1. Nothing is hand-typed.
"""
import glob
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

WS = Path(__file__).parent.parent  # artifact code/ root
PAPER = WS.parent / "revision2"
ORIG = WS / "results" / "real"
REV = ORIG  # D-cell runs ship inside the artifact
OUT = WS / "analysis" / "tables_out"

DS_LABEL = {"diginetica": "Diginetica", "gowalla": "Gowalla",
            "yoochoose_quarter": "Yoo-1/4", "nowplaying": "Nowplaying",
            "retailrocket": "RetailRocket", "yoochoose": "Yoochoose"}
CELLS = ["D0R1", "D0R1nod", "D0R0", "D0R0nod", "D1R1", "D1R1nod",
         "D1R0", "D1R0nod"]
CELL_LABEL = {
    "D0R1": "$L{=}2$, residual, denoise",
    "D0R1nod": "$L{=}2$, residual, no denoise",
    "D0R0": "$L{=}2$, no residual, denoise",
    "D0R0nod": "$L{=}2$, no residual, no denoise",
    "D1R1": "$L{=}1$, residual, denoise",
    "D1R1nod": "$L{=}1$, residual, no denoise",
    "D1R0": "$L{=}1$, no residual, denoise",
    "D1R0nod": "$L{=}1$, no residual, no denoise",
}


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lb + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def paired_t(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    sd = d.std(ddof=1)
    if n < 2 or sd == 0:
        return float("nan"), float("nan")
    t = d.mean() / (sd / math.sqrt(n))
    return t, _betai(0.5 * (n - 1), 0.5, (n - 1) / ((n - 1) + t * t))


def wilcoxon_exact(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    w = r[d > 0].sum()
    tot = n * (n + 1) // 2
    # two-sided: count rank sums at least as extreme in EITHER tail;
    # w_obs = min(w, tot-w) keeps the region {<= w_obs} u {>= tot-w_obs}
    # correct when w itself is the large tail (e.g. all diffs positive)
    w_obs = min(w, tot - w)
    ext = sum(1 for signs in itertools.product([0, 1], repeat=n)
              if (ww := sum(ri for ri, s in zip(r, signs) if s)) <= w_obs
              or ww >= tot - w_obs)
    return ext / 2 ** n


def holm(pv):
    m = len(pv)
    order = sorted(range(m), key=lambda i: pv[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pv[i]))
        adj[i] = run
    return adj


def rec(d, src="rev"):
    return {"hr": d["test"]["HR@20"] * 100, "mrr": d["test"]["MRR@20"] * 100,
            "ndcg": d["test"]["NDCG@20"] * 100, "ep": d.get("best_epoch"),
            "src": src}


def load_cells():
    cells = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(str(ORIG / "seed_*.json"))):
        d = json.load(open(f))
        if d.get("model") != "mplignn":
            continue
        # The factorial/paired tables pool the RELEASED series (seeds 1-4);
        # the seed-5 runs arrived later and belong to the five-seed
        # stability table (tab:seeds) only. Keep the aggregates disjoint,
        # as printed in the paper (same convention as fig3/verify scripts).
        if d["seed"] > 4:
            continue
        a = d.get("ablation") or {}
        cell = ("D0R0" if a.get("no_residual") else "D0R1") + \
               ("nod" if a.get("no_denoise") else "")
        cells[cell][d["dataset"]][d["seed"]] = rec(d, "orig")
    for f in sorted(glob.glob(str(REV / "*.json"))):
        name = Path(f).name
        if name.startswith(("REPRO", "readout", "smoke")):
            continue
        cell = name.split("_")[0]
        if cell not in CELLS + ["LRlowD0R1", "LRlowD1R1"]:
            continue
        d = json.load(open(f))
        seed = int(name.rsplit("_s", 1)[1].split(".")[0])
        cells[cell][d["dataset"]][seed] = rec(d)
    return cells


def stats(cells, cell, ds, key="hr"):
    rows = cells.get(cell, {}).get(ds, {})
    if not rows:
        return None
    a = np.array([v[key] for v in rows.values()])
    return {"n": len(a), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "seeds": sorted(rows)}


def run_epochs(cells, cell, ds):
    """Validation-selected best epoch of every run in a cell,
    seed-ordered; used for the epoch-selection diagnostics."""
    rows = cells.get(cell, {}).get(ds, {})
    return [rows[s]["ep"] for s in sorted(rows)]


def tex(st, nd=2):
    if st is None:
        return "---"
    if st["std"] is None:
        return f"{st['mean']:.{nd}f}$^\\dagger$"
    return f"{st['mean']:.{nd}f}$\\pm${st['std']:.{nd}f}"


def table_factorial(cells, datasets, fh, key, label, caption):
    cols = [d for d in datasets
            if sum(1 for c in CELLS if stats(cells, c, d)) >= 3]
    fh.write(f"\\begin{{table}}[t]\n\\caption{{{caption}}}\n"
             f"\\label{{{label}}}\n\\centering\\small\n"
             "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l " +
             "c" * len(cols) + "}\n\\toprule\n")
    fh.write("Cell & " + " & ".join(DS_LABEL[d] for d in cols) +
             " \\\\\n\\midrule\n")
    for c in CELLS:
        vals = [tex(stats(cells, c, d, key)) for d in cols]
        if all(v == "---" for v in vals):
            continue
        fh.write(f"{CELL_LABEL[c]} & " + " & ".join(vals) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def table_interaction(cells, fh):
    fh.write("\\begin{table}[t]\n\\caption{Denoising effect by residual "
             "configuration (HR@20, \\%), at both local depths. "
             "$\\Delta_{\\mathrm{den}}$ is "
             "denoise-on minus denoise-off at that configuration; the "
             "interaction is $\\Delta_{\\mathrm{den}}(R_1)-"
             "\\Delta_{\\mathrm{den}}(R_0)$, $p_t$ a paired $t$-test and "
             "$p_w$ the exact signed-rank Wilcoxon (floor $.125$ at "
             "$n{=}4$), both over matched seeds. A significant "
             "interaction means the "
             "denoising module has no configuration-free effect.}\n"
             "\\label{tab:interaction}\n\\centering\\small\n"
             "\\setlength{\\tabcolsep}{3pt}\n"
             "\\begin{tabular}{ll cccc cc}\n\\toprule\n"
             "Dataset & & $\\Delta_{\\mathrm{den}}(R_1)$ & "
             "$\\Delta_{\\mathrm{den}}(R_0)$ & interaction & $p_t$ & "
             "$p_w$ & $n$ \\\\\n"
             "\\midrule\n")
    blocks = [("$L{=}2$", ("D0R1", "D0R1nod", "D0R0", "D0R0nod")),
              ("$L{=}1$", ("D1R1", "D1R1nod", "D1R0", "D1R0nod"))]
    for ds in ["diginetica", "gowalla", "yoochoose_quarter"]:
        for bi, (dlab, (c1, c1n, c0, c0n)) in enumerate(blocks):
            name = DS_LABEL[ds] if bi == 0 else ""
            need = {"$R_1$": (c1, c1n), "$R_0$": (c0, c0n)}
            lack = [k for k, (a, b) in need.items()
                    if not (cells.get(a, {}).get(ds)
                            and cells.get(b, {}).get(ds))]
            if lack:
                fh.write(f"{name} & {dlab} & \\multicolumn{{6}}{{c}}{{"
                         f"incomplete: {', '.join(lack)} on/off pair "
                         f"not run}} \\\\\n")
                continue
            d1, d1n = cells[c1][ds], cells[c1n][ds]
            d0, d0n = cells[c0][ds], cells[c0n][ds]
            common = sorted(set(d1) & set(d1n) & set(d0) & set(d0n))
            if len(common) < 2:
                fh.write(f"{name} & {dlab} & \\multicolumn{{6}}{{c}}{{"
                         f"fewer than two matched seeds}} \\\\\n")
                continue
            x1 = np.array([d1[s]["hr"] - d1n[s]["hr"] for s in common])
            x0 = np.array([d0[s]["hr"] - d0n[s]["hr"] for s in common])
            _, p = paired_t(x1, x0)
            pw = wilcoxon_exact(x1, x0)
            fh.write(f"{name} & {dlab} & {x1.mean():+.2f} & {x0.mean():+.2f} &"
                     f" {(x1 - x0).mean():+.2f} & {p:.3f} & {pw:.3f} & "
                     f"{len(common)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def table_optim(cells, fh):
    fh.write("\\begin{table}[t]\n\\caption{The depth comparison at two "
             "step sizes (HR@20, \\%). Each row pairs cells that differ "
             "only in local depth (deep $L{=}2$ vs shallow $L{=}1$) at the "
             "same residual setting; the low-LR rows retrain the identical "
             "architectures with $\\mathrm{lr}=5\\times10^{-5}$. ``Best "
             "epochs'' lists the validation-selected epoch of each run "
             "in the cell, so a model selecting epoch~1 never improved "
             "on its first epoch.}\n\\label{tab:optim}\n\\centering\\small\n"
             "\\setlength{\\tabcolsep}{3pt}\n"
             "\\begin{tabular}{@{}p{1.35cm}p{2.0cm}cc"
             ">{\\raggedright\\arraybackslash}p{2.3cm}@{}}\n"
             "\\toprule\nDataset & Pairing & deep & shallow & best epochs "
             "\\\\\n\\midrule\n")
    pairs = [("lr $5{\\times}10^{-4}$, $R_0$", "D0R0", "D1R0"),
             ("lr $5{\\times}10^{-4}$, $R_1$", "D0R1", "D1R1"),
             ("lr $5{\\times}10^{-5}$, $R_1$", "LRlowD0R1", "LRlowD1R1")]
    for ds in ["diginetica", "gowalla"]:
        for lab, cd, cs in pairs:
            a, b = stats(cells, cd, ds), stats(cells, cs, ds)
            if not a or not b:
                fh.write(f"{DS_LABEL[ds]} & {lab} & --- & --- & --- \\\\\n")
                continue
            fh.write(f"{DS_LABEL[ds]} & {lab} & {tex(a)} & {tex(b)} & "
                     f"{run_epochs(cells, cd, ds)} vs "
                     f"{run_epochs(cells, cs, ds)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def table_main(cells, fh):
    base_models = ["itemknn", "gru4rec", "narm", "srgnn", "lessr", "gcegnn",
                   "mtd", "dsan", "msgifsr", "msgat"]
    pretty = {"itemknn": "Item-KNN", "gru4rec": "GRU4Rec", "narm": "NARM",
              "srgnn": "SR-GNN", "lessr": "LESSR", "gcegnn": "GCE-GNN",
              "mtd": "MTD", "dsan": "DSAN", "msgifsr": "MSGIFSR",
              "msgat": "MSGAT"}
    dss = ["diginetica", "gowalla", "yoochoose_quarter", "nowplaying",
           "retailrocket"]
    seedav = defaultdict(lambda: defaultdict(list))
    single = defaultdict(dict)
    for f in sorted(glob.glob(str(ORIG / "seed_*.json"))):
        d = json.load(open(f))
        seedav[d["model"]][d["dataset"]].append(
            (d["test"]["HR@20"] * 100, d["test"]["MRR@20"] * 100))
    for m in base_models:
        for ds in dss:
            p = ORIG / f"{m}_{ds}.json"
            if p.exists():
                d = json.load(open(p))
                single[m][ds] = (d["test"]["HR@20"] * 100,
                                 d["test"]["MRR@20"] * 100)
    fh.write("\\begin{table*}[t]\n\\caption{Same-budget comparison "
             "(HR@20 / MRR@20, \\%), every cell produced by the released "
             "pipeline under the budget of Section~\\ref{sec:setup}. "
             "Mean$\\pm$std over released seeds where more than one run "
             "exists, $^\\dagger$ for a single run. Single runs support no "
             "difference claim, and the table is a same-budget reference, "
             "not a ranking of methods (Section~\\ref{sec:overall}).}\n"
             "\\label{tab:main}\n\\centering\\small\n"
             "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l " +
             " ".join(["cc"] * len(dss)) + "}\n\\toprule\n")
    fh.write("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}"
                               for d in dss) + " \\\\\n")
    fh.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
             "\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n")
    fh.write("Method & " + " & ".join(["HR & MRR"] * len(dss)) +
             " \\\\\n\\midrule\n")
    for m in base_models:
        seen, txt = False, []
        for ds in dss:
            if seedav.get(m, {}).get(ds):
                a = np.array(seedav[m][ds])
                txt += [f"{a[:,0].mean():.2f}$\\pm${a[:,0].std(ddof=1):.2f}",
                        f"{a[:,1].mean():.2f}$\\pm${a[:,1].std(ddof=1):.2f}"]
                seen = True
            elif m in single and ds in single[m]:
                h, r = single[m][ds]
                txt += [f"{h:.2f}$^\\dagger$", f"{r:.2f}$^\\dagger$"]
                seen = True
            else:
                txt += ["---", "---"]
        if seen:
            fh.write(f"{pretty[m]} & " + " & ".join(txt) + " \\\\\n")
    fh.write("\\midrule\n")
    for c in ["D0R1", "D0R0", "D1R1", "D1R0"]:
        txt = []
        for ds in dss:
            txt += [tex(stats(cells, c, ds, "hr")),
                    tex(stats(cells, c, ds, "mrr"))]
        if all(v == "---" for v in txt):
            continue
        depth = "L=2" if c.startswith("D0") else "L=1"
        res = "residual" if "R1" in c else "no residual"
        fh.write(f"MPLI-GNN ({depth}, {res}) & " + " & ".join(txt) +
                 " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")


def table_paired(cells, fh):
    # diff column = first cell minus second, over matched seeds. The depth
    # contrast is computable only at R_0: the D0 cells live on the released
    # seed series (1-4) while D1R1 was run on 7,13,21,34,42, so D1R1 vs
    # D0R1 shares no seed; D1R0 was run on both series precisely to close
    # that gap.
    comps = [("denoise on $-$ off", "D0R1", "D0R1nod"),
             ("denoise on $-$ off", "D0R0", "D0R0nod"),
             ("denoise on $-$ off", "D1R1", "D1R1nod"),
             ("denoise on $-$ off", "D1R0", "D1R0nod"),
             ("residual on $-$ off", "D0R1", "D0R0"),
             ("residual on $-$ off", "D1R1", "D1R0"),
             ("depth deep $-$ shallow", "D0R0", "D1R0")]
    rows, raw = [], []
    for lab, ca, cb in comps:
        for ds in ["diginetica", "gowalla", "yoochoose_quarter"]:
            A = cells.get(ca, {}).get(ds, {})
            B = cells.get(cb, {}).get(ds, {})
            common = sorted(set(A) & set(B))
            if len(common) < 3:
                continue
            x = np.array([A[s]["hr"] for s in common])
            y = np.array([B[s]["hr"] for s in common])
            t, p = paired_t(x, y)
            raw.append(p)
            rows.append((lab, ds, len(common), (x - y).mean(), t, p,
                         wilcoxon_exact(x, y)))
    fh.write("\\begin{table}[t]\n\\caption{Paired comparisons on cells with "
             "at least three matched seeds (HR@20). $p$ is a two-sided "
             "paired $t$-test over the matched seeds and "
             "$p_{\\mathrm{Holm}}$ corrects across table rows. We lead with "
             "$t$ because the exact Wilcoxon test cannot attain "
             "$p<0.125$ at $n{=}4$; both are computed by the released "
             "analysis script.}\n"
             "\\label{tab:paired}\n\\centering\\small\n"
             "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{ll r rr r}\n"
             "\\toprule\nComparison & Dataset & $n$ & diff & $p$ & "
             "$p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
    if not rows:
        fh.write("\\multicolumn{6}{c}{no cells with matched seeds yet} \\\\\n")
    for (lab, ds, n, d, t, p, wp), pa in zip(rows, holm(raw) if raw else []):
        star = "$^{*}$" if pa < 0.05 else ""
        fh.write(f"{lab} & {DS_LABEL[ds]} & {n} & {d:+.2f}{star} & "
                 f"{p:.3f} & {pa:.3f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print("=== loaded ===")
    for c in CELLS + ["LRlowD0R1", "LRlowD1R1"]:
        for ds in sorted(cells.get(c, {})):
            st = stats(cells, c, ds)
            print(f"  {c:10s} {ds:18s} n={st['n']:2d} "
                  f"HR={st['mean']:6.2f}"
                  + (f"±{st['std']:.2f}" if st['std'] else " (single)"))
    with open(OUT / "generated.tex", "w") as fh:
        fh.write("%% AUTO-GENERATED by make_tables.py --- do not edit.\n\n")
        table_factorial(cells, ["diginetica", "gowalla", "yoochoose_quarter"],
                        fh, "hr", "tab:factorial",
                        "The controlled factorial: mean$\\pm$std HR@20 "
                        "(\\%) over released seeds. The two factors---local "
                        "depth and the denoising path---are toggled "
                        "independently; $R_0$/$R_1$ mark the absence/presence "
                        "of the residual, and ``---'' marks a cell that has "
                        "not been run. $^\\dagger$ marks a single run.")
        table_factorial(cells, ["diginetica", "gowalla", "yoochoose_quarter"],
                        fh, "mrr", "tab:factorial_mrr",
                        "The factorial of Table~\\ref{tab:factorial} "
                        "reporting MRR@20 (\\%).")
        table_interaction(cells, fh)
        table_optim(cells, fh)
        table_main(cells, fh)
        table_paired(cells, fh)
    print(f"\nwrote {OUT/'generated.tex'}")


if __name__ == "__main__":
    main()
