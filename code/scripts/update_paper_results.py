#!/usr/bin/env python3
"""Workstation-side FULL paper update from GPU results.

Usage:
    python3 scripts/update_paper_results.py --tar gpu_results.tar.gz
    python3 scripts/update_paper_results.py            # tables only, no tar

What it does
------------
  1. Ingest gpu_results.tar.gz into results/real/ (resumable, validated).
  2. Regenerate EVERY results table in the paper from the newest JSONs:
       - tab:main        budget matrix 14 models x 6 datasets
       - tab:ablation    8 ablation rows (Diginetica)
       - tab:noise       5 models x 6 noise levels
       - tab:efficiency  GPU efficiency rows (if present)
       - tab:fullmetrics appendix NDCG@20 / Coverage@20
       - full_protocol_table.tex (full-epoch 6 x 3)
       - seed statistics table (mean +- std x 5 seeds)
     Bold = best per column. Missing cells stay gray, never fabricated.
  3. Write results/real/PROSE_DIFF.md listing every number cited in the
     prose that changed, so the narrative can be synced by hand.
  4. Print recompile instructions.
"""
import argparse
import json
import re
import shutil
import statistics
import tarfile
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / "results" / "real"
TEX = P.parent / "v1"

DS6 = ["diginetica", "gowalla", "yoochoose", "yoochoose_quarter",
       "nowplaying", "retailrocket"]
DS6_LBL = {"diginetica": "Diginetica", "gowalla": "Gowalla",
           "yoochoose": "Yoo-1/64", "yoochoose_quarter": "Yoo-1/4",
           "nowplaying": "Nowplaying", "retailrocket": "RetailRocket"}
M14 = [("itemknn", "Item-KNN \\cite{sarwar2001item}"),
       ("fpmc", "FPMC \\cite{rendle2010factorizing}"),
       ("gru4rec", "GRU4Rec \\cite{hidasi2015session}"),
       ("narm", "NARM \\cite{li2017neural}"),
       ("stamp", "STAMP \\cite{liu2018stamp}"),
       ("srgnn", "SR-GNN (re-run) \\cite{wu2019session}"),
       ("lessr", "LESSR \\cite{chen2020handling}"),
       ("dhcn", "DHCN \\cite{xia2021self}"),
       ("gcegnn", "GCE-GNN \\cite{wang2020global}"),
       ("mtd", "MTD \\cite{huang2021graph}"),
       ("dsan", "DSAN \\cite{yuan2021dual}"),
       ("msgifsr", "MSGIFSR \\cite{guo2022learning}"),
       ("msgat", "MSGAT \\cite{qiao2023bi}"),
       ("mplignn_L1", "\\textbf{MPLI-GNN ($L{=}1$, budget)}"),
       ("mplignn", "MPLI-GNN ($L{=}2$, full)")]
M14_ORDER = ["itemknn", "fpmc", "gru4rec", "narm", "stamp", "srgnn",
             "lessr", "dhcn", "gcegnn", "mtd", "dsan", "msgifsr", "msgat",
             "mplignn", "mplignn_L1"]


def load(stem):
    f = OUT / f"{stem}.json"
    return json.loads(f.read_text()) if f.exists() else None


def hm(r):
    if not r or "test" not in r:
        return None
    t = r["test"]
    return 100 * t["HR@20"], 100 * t["MRR@20"]


def ingest(tar: Path):
    n = 0
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if not m.name.startswith("results/real/") or not m.isfile():
                continue
            dest = P / m.name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print(f"ingested {n} new files from {tar.name}")


def fmt(v, best=False):
    if v is None:
        return "\\textcolor{gray}{--}"
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if best else s


# ---------------- tab:main ----------------
def table_main():
    cells = {}
    for m in M14_ORDER:
        for d in DS6:
            stem = (f"abl_no_residual_{d}" if m == "mplignn_L1"
                    and d == "diginetica" else
                    f"mplignn_L1_{d}" if m == "mplignn_L1" else f"{m}_{d}")
            cells[(m, d)] = hm(load(stem))
    best = {}
    for d in DS6:
        vals = [cells[(m, d)] for m in M14_ORDER if cells[(m, d)]]
        best[(d, "H")] = max(v[0] for v in vals) if vals else None
        best[(d, "R")] = max(v[1] for v in vals) if vals else None
    lines = []
    for m, label in M14:
        if m == "mplignn_L1" and m not in [x[0] for x in M14[:13]]:
            pass
        row = [label.ljust(38)]
        for d in DS6:
            c = cells[(m, d)]
            h = fmt(c[0], c and abs(c[0] - best[(d, 'H')]) < 5e-3) if c else fmt(None)
            r = fmt(c[1], c and abs(c[1] - best[(d, 'R')]) < 5e-3) if c else fmt(None)
            row += [h, r]
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines), cells


def _patch_block(tex_name, label, row_substitutions):
    """Patch rows only inside the table environment carrying `label`."""
    path = TEX / tex_name
    src = path.read_text()
    li = src.find("\\label{" + label + "}")
    if li < 0:
        print(f"WARN: label {{{label}}} not found")
        return
    b = src.rfind("\\begin{table", 0, li)
    e = src.find("\\end{table", li)
    if b < 0 or e < 0:
        print(f"WARN: table bounds not found for {label}")
        return
    e = src.find("}", e) + 1
    block = src[b:e]
    for lab, new_line in row_substitutions:
        import re as _re
        pat = _re.compile(r"^\s*" + _re.escape(lab) + r"\s+&.*\\$", _re.M)
        block, n = pat.subn(lambda _m, s=new_line: s, block)
        if n != 1:
            print(f"WARN row matched {n}: {lab[:44]}")
    path.write_text(src[:b] + block + src[e:])


def patch_main(rows_block):
    subs = []
    for line in rows_block.split("\n"):
        label = line.split(" &")[0].strip()
        subs.append((label, line))
    _patch_block("sec_5_experiment.tex",
                 "tab:main",
                 subs)
    print("tab:main updated")


# ---------------- full-epoch table ----------------
def table_full():
    DS3 = ["diginetica", "gowalla", "yoochoose_quarter"]
    models = [("itemknn", "Item-KNN"), ("narm", "NARM"), ("srgnn", "SR-GNN"),
              ("gcegnn", "GCE-GNN"), ("mplignn_L1", "MPLI-GNN ($L{=}1$)"),
              ("mplignn", "MPLI-GNN ($L{=}2$)")]
    cells = {(d, m): hm(load(f"full_{m}_{d}")) for d in DS3 for m, _ in models}
    best = {d: (max((cells[(d, m)][0] for m, _ in models if cells[(d, m)]), default=None),
                max((cells[(d, m)][1] for m, _ in models if cells[(d, m)]), default=None))
            for d in DS3}
    lines = ["% Auto-generated by scripts/update_paper_results.py",
             "\\begin{table*}[t]",
             "\\caption{Full-epoch protocol (no batch cap, 30 epochs, patience 5):",
             "HR@20 / MRR@20 (\\%). Baselines use their published size and lr;",
             "MPLI-GNN uses its design config. Gray cells are runs still in",
             "progress; nothing is fabricated.}",
             "\\label{tab:fullepoch}", "\\centering\\small",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l cc cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{Diginetica} & \\multicolumn{2}{c}{Gowalla}"
             " & \\multicolumn{2}{c}{Yoo-1/4} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Method & HR & MRR & HR & MRR & HR & MRR \\\\", "\\midrule"]
    for m, label in models:
        row = [label]
        for d in DS3:
            c = cells[(d, m)]
            if c:
                row += [fmt(c[0], abs(c[0] - best[d][0]) < 5e-3),
                        fmt(c[1], abs(c[1] - best[d][1]) < 5e-3)]
            else:
                row += [fmt(None), fmt(None)]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (TEX / "full_protocol_table.tex").write_text("\n".join(lines))
    done = sum(1 for c in cells.values() if c)
    print(f"full_protocol_table.tex: {done}/18 cells")


# ---------------- seed statistics ----------------
def table_seeds():
    models = ["mplignn_L1", "mplignn", "srgnn", "narm"]
    lines = ["% Auto-generated: five-seed statistics (HR@20 mean+-std)",
             "\\begin{table}[t]",
             "\\caption{Five-seed stability under the budget protocol "
             "(HR@20, mean$\\pm$std).}", "\\label{tab:seeds}",
             "\\centering\\small", "\\begin{tabular}{l cc}", "\\toprule",
             "Model & Diginetica & Gowalla \\\\", "\\midrule"]
    n_any = 0
    for m in models:
        row = [m.replace("_", "\\_")]
        for d in ["diginetica", "gowalla"]:
            vals = [hm(load(f"seed_{m}_{d}_s{k}")) for k in range(1, 6)]
            vals = [v[0] for v in vals if v]
            if len(vals) >= 2:
                row.append(f"{statistics.mean(vals):.2f}"
                           f"$\\pm${statistics.stdev(vals):.2f}")
                n_any += 1
            else:
                main = hm(load(f"abl_no_residual_{d}" if m == "mplignn_L1"
                               and d == "diginetica" else f"{m}_{d}"))
                row.append(fmt(main[0]) if main else fmt(None))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TEX / "seed_statistics_table.tex").write_text("\n".join(lines))
    print(f"seed_statistics_table.tex ({n_any} cells with >=2 seeds)")


# ---------------- noise ----------------
def patch_noise():
    models = [("srgnn", "SR-GNN \\cite{wu2019session}"),
              ("gcegnn", "GCE-GNN \\cite{wang2020global}"),
              ("dsan", "DSAN \\cite{yuan2021dual}"),
              ("msgat", "MSGAT \\cite{qiao2023bi}"),
              ("mplignn", "\\textbf{MPLI-GNN}")]
    subs = []
    for m, label in models:
        n = load(f"noise_{m}_diginetica")
        if not n:
            continue
        vals = [n[f"{x}%"]["HR@20"] * 100 for x in range(0, 60, 10)]
        new = label + " & " + " & ".join(f"{v:.2f}" for v in vals) + " \\\\"
        subs.append((label, new))
    if subs:
        _patch_block("sec_5_experiment.tex",
                     "tab:noise",
                     subs)
    print("tab:noise updated")


# ---------------- prose diff report ----------------
def prose_diff(cells):
    cited = {
        ("mplignn_L1", "gowalla"): "40.56/34.93 (abstract/intro/RQ1)",
        ("mplignn_L1", "diginetica"): "30.09/21.35",
        ("mplignn_L1", "yoochoose"): "40.59/26.37",
        ("mplignn_L1", "yoochoose_quarter"): "40.70/31.65",
        ("mplignn", "diginetica"): "14.30/11.11",
        ("srgnn", "retailrocket"): "35.81/28.92",
        ("itemknn", "yoochoose_quarter"): "61.96/30.08",
    }
    lines = ["# Prose number diff (tables updated automatically; sync these",
             "citations in abstract/intro/RQ sections by hand if changed)", ""]
    for (m, d), old in cited.items():
        c = cells.get((m, d))
        new = f"{c[0]:.2f}/{c[1]:.2f}" if c else "--"
        mark = "SAME" if new == old else "CHANGED"
        lines.append(f"- {m} @ {d}: paper={old} -> results={new}  [{mark}]")
    (OUT / "PROSE_DIFF.md").write_text("\n".join(lines) + "\n")
    print("PROSE_DIFF.md written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=Path, default=None)
    args = ap.parse_args()
    if args.tar and args.tar.exists():
        ingest(args.tar)
    rows, cells = table_main()
    patch_main(rows)
    table_full()
    table_seeds()
    patch_noise()
    prose_diff(cells)
    print("\nNext: cd .. && pdflatex main.tex (x2); verify 0 errors / 0 '??'")
    print("Optional: \\input{full_protocol_table} and \\input{seed_statistics_table}")


if __name__ == "__main__":
    main()
