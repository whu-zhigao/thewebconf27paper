#!/usr/bin/env python3
#Aggregate results/real/*.json into a markdown report and the mapping.json
# consumed by fill_tables.py.
import json
from pathlib import Path

REAL = Path("results/real")
DATASETS = ["diginetica", "gowalla", "yoochoose", "nowplaying", "retailrocket"]


def load(name):
    p = REAL / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}" if x is not None else None


def main():
    # ---- main table cells (row-major: HR, MRR per dataset) --------------
    order = ["diginetica", "gowalla", "yoochoose", "nowplaying", "retailrocket"]
    main_cells = {"itemknn": [], "srgnn": [], "mplignn": []}
    for model in main_cells:
        for d in order:
            r = load(f"{model}_{d}.json")
            if r and "test" in r:
                main_cells[model] += [pct(r["test"]["HR@20"]),
                                      pct(r["test"]["MRR@20"])]
            else:
                main_cells[model] += ["--", "--"]

    mapping = {"tab:main": {"cells": main_cells["itemknn"] +
                            ["--"] * 20 +          # FPMC..NARM..STAMP rows
                            main_cells["srgnn"] + ["--"] * 16 +
                            main_cells["mplignn"]}}

    # ---- ablation table (Diginetica, Gowalla, Yoo, Nowplaying) -----------
    abl_rows = {"full": "mplignn_diginetica",
                "no_denoise": "abl_no_denoise_diginetica",
                "softmax": "abl_softmax_attn_diginetica",
                "no_anchor": "abl_no_anchor_diginetica",
                "no_global": "abl_no_global_diginetica"}
    abl_cells = []
    for _, f in abl_rows.items():
        r = load(f + ".json") if not f.endswith(".json") else load(f)
        if r and "test" in r:
            abl_cells += [pct(r["test"]["HR@20"]), pct(r["test"]["MRR@20"])]
        else:
            abl_cells += ["--", "--"]
    # remaining variants are pending; fill with placeholders
    while len(abl_cells) < 7 * 2 * 4:
        abl_cells.append("--")
    mapping["tab:ablation"] = {"cells": abl_cells}

    # ---- noise table ------------------------------------------------------
    noise = load("noise_diginetica.json")
    noise_cells = ["--"] * 6 * 4          # 4 baseline rows pending
    if noise:
        for x in ["0%", "10%", "20%", "30%", "40%", "50%"]:
            noise_cells += [pct(noise[x]["HR@20"])]
    else:
        noise_cells += ["--"] * 6
    mapping["tab:noise"] = {"cells": noise_cells}

    (REAL / "mapping.json").write_text(json.dumps(mapping, indent=1))

    # ---- markdown summary --------------------------------------------------
    lines = ["# Real-Data Experiment Summary", "",
             f"Protocol: CPU, d=64, 8 epochs x <=300 batches, batch 128, "
             f"early stop patience 3.", "",
             "## Pilot study (Section 5.1)", "",
             "| dataset | P1 multi-intent | P2 shift frac | P2 HR stable | "
             "P2 HR shift | P2 gap | P3 local dom |", "|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        p = load(f"pilot_{d}.json")
        if p:
            lines.append(f"| {d} | {pct(p['P1_multi_intent_fraction'],1)}% | "
                         f"{pct(p['P2_shift_fraction'],1)}% | "
                         f"{pct(p['P2_hr_stable'],1)}% | "
                         f"{pct(p['P2_hr_shift'],1)}% | "
                         f"{pct(p['P2_hr_gap'],1)}pp | "
                         f"{pct(p['P3_local_dominance_fraction'],1)}% |")
    lines += ["", "## Main results (RQ1, HR@20 / MRR@20 %)", "",
              "| model | " + " | ".join(DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)]
    for model, label in [("itemknn", "Item-KNN"), ("srgnn", "SR-GNN-lite"),
                         ("mplignn", "MPLI-GNN")]:
        row = []
        for d in DATASETS:
            r = load(f"{model}_{d}.json")
            row.append(f"{pct(r['test']['HR@20'])}/{pct(r['test']['MRR@20'])}"
                       if r and "test" in r else "--")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines += ["", "## Ablations on Diginetica (RQ2)", "",
              "| variant | HR@20 | MRR@20 |", "|---|---|---|"]
    for label, f in abl_rows.items():
        r = load(f)
        if r and "test" in r:
            lines.append(f"| {label} | {pct(r['test']['HR@20'])} | "
                         f"{pct(r['test']['MRR@20'])} |")
    (REAL / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REAL / 'mapping.json'} and summary.md")


if __name__ == "__main__":
    main()
