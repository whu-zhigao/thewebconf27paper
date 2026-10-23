#!/usr/bin/env python3
"""Runs manifest: per factorial cell, how many released runs exist and
from which series (v1 released seed_* records vs. revision run_cells.py
records). Backs the n-per-cell column of Table 2 and the manifest table
of Appendix "In-flight runs" (item M6 of the revision): every paper
number is traceable to a count of records on disk.

Usage:  python3 analysis/runs_manifest.py            (prints a LaTeX tabular)
        python3 analysis/runs_manifest.py --ixseed   (per-seed interaction
                                                      components; backs
                                                      tab:ixseed)
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
_ARGS = sys.argv[1:]              # capture before the import shield below
spec = importlib.util.spec_from_file_location("mt", CODE / "analysis" / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]          # shield mt from our CLI flags
spec.loader.exec_module(mt)

CELL_ORDER = ["D0R1", "D0R1nod", "D0R0", "D0R0nod",
              "D1R1", "D1R1nod", "D1R0", "D1R0nod"]
CELL_LABEL = {
    "D0R1": "$L{=}2$, residual", "D0R1nod": "$L{=}2$, residual",
    "D0R0": "$L{=}2$, no residual", "D0R0nod": "$L{=}2$, no residual",
    "D1R1": "$L{=}1$, residual", "D1R1nod": "$L{=}1$, residual",
    "D1R0": "$L{=}1$, no residual", "D1R0nod": "$L{=}1$, no residual",
}
DATASETS = ["diginetica", "gowalla", "yoochoose_quarter"]


def load_cells_with_src():
    """Same records as make_tables.load_cells, keeping the series tag."""
    import glob
    import json
    cells = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(str(mt.ORIG / "seed_*.json"))):
        d = json.load(open(f))
        if d.get("model") != "mplignn":
            continue
        # same convention as make_tables.load_cells: the factorial pools
        # the released series (seeds 1-4); seed 5 belongs to tab:seeds only
        if d["seed"] > 4:
            continue
        a = d.get("ablation") or {}
        cell = ("D0R0" if a.get("no_residual") else "D0R1") + \
               ("nod" if a.get("no_denoise") else "")
        cells[cell][d["dataset"]][d["seed"]] = mt.rec(d, "orig")
    for f in sorted(glob.glob(str(mt.REV / "*.json"))):
        name = Path(f).name
        if name.startswith(("REPRO", "readout", "smoke")):
            continue
        cell = name.split("_")[0]
        if cell not in mt.CELLS + ["LRlowD0R1", "LRlowD1R1"]:
            continue
        d = json.load(open(f))
        seed = int(name.rsplit("_s", 1)[1].split(".")[0])
        cells[cell][d["dataset"]][seed] = mt.rec(d)
    return cells


def main():
    if "--ixseed" in _ARGS:
        return ixseed()
    cells = load_cells_with_src()
    print(r"\begin{tabular}{l ccc}")
    print(r"\toprule")
    print(r"Cell (den.\ on / off) & Diginetica & Gowalla & Yoo-1/4 \\")
    print(r"\midrule")
    for cell in CELL_ORDER:
        den = "off" if cell.endswith("nod") else "on"
        counts = []
        for ds in DATASETS:
            recs = cells.get(cell, {}).get(ds, {})
            n_orig = sum(1 for v in recs.values() if v["src"] == "orig")
            n_new = sum(1 for v in recs.values() if v["src"] == "rev")
            # every entry prints the full "n (k+m)" so the notation the
            # caption defines matches every row (was "n (k)" when m=0)
            counts.append(f"{len(recs)} ({n_orig}+{n_new})")
        print(" & ".join([f"{CELL_LABEL[cell]} / {den}"] + counts) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    total = sum(len(cells.get(c, {}).get(ds, {}))
                for c in CELL_ORDER for ds in DATASETS)
    print(f"% total mplignn budget-protocol records in this table: {total}",
          file=sys.stderr)


def ixseed():
    """Per-seed components of the denoise x residual interaction
    (tab:ixseed): for each depth and dataset, over the seeds shared by
    both denoise contrasts, the paired denoise on-off difference within
    each residual slice and their difference (the interaction)."""
    cells = load_cells_with_src()
    for ds in ["diginetica", "gowalla"]:
        for L, on, off, on2, off2 in [
                ("$L{=}2$", "D0R1", "D0R1nod", "D0R0", "D0R0nod"),
                ("$L{=}1$", "D1R1", "D1R1nod", "D1R0", "D1R0nod")]:
            common = sorted(set(cells[on][ds]) & set(cells[off][ds])
                            & set(cells[on2][ds]) & set(cells[off2][ds]))
            rows = []
            for s in common:
                d_res = cells[on][ds][s]["hr"] - cells[off][ds][s]["hr"]
                d_nores = cells[on2][ds][s]["hr"] - cells[off2][ds][s]["hr"]
                rows.append((s, d_res, d_nores, d_res - d_nores))
            fmt = lambda v: f"{v:+.2f}"
            seeds = "{" + ",".join(str(s) for s in common) + "}"
            print(f"{ds} {L} seeds {seeds}")
            for s, a, b, c in rows:
                print(f"  s{s:2d}: d_res {fmt(a)}  d_nores {fmt(b)}  ix {fmt(c)}")
            rng = lambda xs: f"{fmt(min(xs))} to {fmt(max(xs))}"
            print(f"  summary: d_res {rng([r[1] for r in rows])} | "
                  f"d_nores {rng([r[2] for r in rows])} | "
                  f"ix {rng([r[3] for r in rows])}")


if __name__ == "__main__":
    main()
