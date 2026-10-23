#!/usr/bin/env python3
"""Populate the paper's placeholder tables from result JSON files.

Every pending cell in the LaTeX source is a ``\\pmark`` macro.  This tool
replaces them, table by table, with numbers from the JSON produced by the
training/analysis scripts.  Tables are identified by their
``\\label{tab:...}`` and cells are filled row-major.

Usage:
  python fill_tables.py --mapping results/mapping.json --tex ../v1/sec_5_experiment.tex

mapping.json schema (all values are strings already formatted for LaTeX):
  {"tab:main": {"rows": 14, "cells": ["50.21", "18.07", ...]},
   "tab:ablation": {...}}
"""
import argparse
import json
import re
import shutil
from pathlib import Path


def fill_table(tex: str, label: str, cells, dry_run=False):
    """Replace successive \\pmark inside the table carrying ``label``."""
    pat = re.compile(r"(\\begin\{table\*?\}.*?\\label\{" + re.escape(label)
                     + r"\}.*?\\end\{table\*?\})", re.S)
    m = pat.search(tex)
    if not m:
        raise SystemExit(f"label {label} not found in tex")
    block = m.group(1)
    it = iter(cells)

    def _sub(_):
        try:
            return next(it)
        except StopIteration:
            raise SystemExit(f"not enough cells provided for {label}")

    new_block, n = re.subn(r"\\pmark", _sub, block)
    leftover = len(list(it))
    if leftover:
        print(f"warning: {leftover} unused cells for {label}")
    print(f"{label}: filled {n} cells")
    return tex[:m.start()] + new_block + tex[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    mapping = json.loads(args.mapping.read_text())
    tex = args.tex.read_text()
    for label, spec in mapping.items():
        tex = fill_table(tex, label, spec["cells"])
    if args.in_place:
        backup = args.tex.with_suffix(".tex.bak")
        shutil.copy(args.tex, backup)
        args.tex.write_text(tex)
        print(f"wrote {args.tex} (backup at {backup})")
    else:
        out = args.tex.with_suffix(".filled.tex")
        out.write_text(tex)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
