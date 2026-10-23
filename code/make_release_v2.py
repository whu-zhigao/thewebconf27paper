#!/usr/bin/env python3
"""Build release_v2: the complete, auditable artifact for revision 2 of
the paper. Supersedes make_release.py (which predates the factorial D-cell
runs, the five-seed series, the sensitivity sweeps, the full-epoch
protocol, and the analysis/ tooling). Deterministic; re-run any time.

Contents:
  code/     library + scripts + analysis tools + tests + drivers + all
            result records (JSON) + re-scoring checkpoints
  paper/    revision-2 LaTeX sources + compiled PDF + class files
  ARTIFACT_GUIDE.md, MANIFEST.json, SHA256SUMS

Excluded by design: raw datasets and derived pickles (licenses; pinned by
split_fingerprints.json instead), factorial-cell checkpoints (JSON records
complete; size), author-internal notes (REPORT.md, REBUTTAL_PLAYBOOK.md).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

P = Path(__file__).parent
PAPER = P.parent  # code/ and all paper versions are siblings under www/
REL = PAPER / "release_v2"      # overridden by --lite

CODE_FILES = [  # single files at code/ root
    "README.md", "NOTES.md", "GPU_RUNBOOK.md", "split_fingerprints.json",
    "run_baselines.sh", "run_smoke.sh", "run_real.sh", "run_fullbudget.py",
    "run_fullbudget_seeds.py", "run_gpu.py", "run_all_gpu.py",
    "run_gpu_yoochoose.py", "make_gpu_package.py", "make_release_v2.py",
    "fill_tables.py", "local_patch.sh",
    "gpu.log", "gpu2.log", "gpu_all.log", "gpu_run.log", "discriminating.log",
]
CODE_DIRS = ["mplignn", "scripts", "tests", "analysis", "results/real"]
# internal working notes inside results/real (author-facing only)
EXCLUDE_NAMES = {"REPORT.md", "PROSE_DIFF.md", "summary.md"}
PAPER_FILES = [  # relative to the paper directory
    "revision2/ARTIFACT_GUIDE.md",
    "revision2/main_r2.tex", "revision2/ref.bib",
    "revision2/acmart.cls", "revision2/ACM-Reference-Format.bst",
    "revision2/sec_1_intro_r2.tex", "revision2/sec_2_related_r2.tex",
    "revision2/sec_3_problem_r2.tex", "revision2/sec_4_method_r2.tex",
    "revision2/sec_5_experiment_r2.tex", "revision2/sec_appendix_r2.tex",
]
PAPER_PDF = "revision2/main_r2.pdf"
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", "tables_out"}
EXCLUDE_SUFFIX = {".pyc", ".DS_Store"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true",
                    help="omit the 212 .pt checkpoints (~2 GB) and the "
                         "compiled PDF; keep every JSON record, all code, "
                         "and the LaTeX sources. The 222-check verification "
                         "runs unchanged.")
    args = ap.parse_args()
    global REL
    rel_name = "release_v2_lite" if args.lite else "release_v2"
    REL = PAPER / rel_name
    if REL.exists():
        shutil.rmtree(REL)
    (REL / "code").mkdir(parents=True)
    copied = []

    for top in CODE_DIRS:
        src = P / top
        if not src.exists():
            print("warn: missing", top)
            continue
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            if any(d in f.parts for d in EXCLUDE_DIRS):
                continue
            if f.suffix in EXCLUDE_SUFFIX or f.name in EXCLUDE_NAMES:
                continue
            if args.lite and f.suffix == ".pt":
                continue
            dst = REL / "code" / f.relative_to(P)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            copied.append(dst)

    for name in CODE_FILES:
        src = P / name
        if not src.exists():
            print("warn: missing", name)
            continue
        dst = REL / "code" / name
        dst.write_bytes(src.read_bytes())
        copied.append(dst)

    for rel in PAPER_FILES:
        src = PAPER / rel
        if not src.exists():
            print("warn: missing", rel)
            continue
        dst = REL / "paper" / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append(dst)
    if not args.lite:  # the compiled PDF ships with the full variant only;
        dst = REL / "paper" / "main_r2.pdf"   # lite keeps the LaTeX sources
        dst.write_bytes((PAPER / PAPER_PDF).read_bytes())
        copied.append(dst)

    # editable figure scripts + rendered PDFs: the paper's \includegraphics
    # targets live in figures_editable/, so the released sources must ship it
    figdir = PAPER / "revision2" / "figures_editable"
    for f in sorted(figdir.glob("*")):
        if f.suffix not in {".py", ".pdf", ".tex", ".md"}:
            continue
        dst = REL / "paper" / "figures_editable" / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f.read_bytes())
        copied.append(dst)

    # the guide also lives at the archive root (entry point)
    dst = REL / "ARTIFACT_GUIDE.md"
    dst.write_bytes((PAPER / "revision2" / "ARTIFACT_GUIDE.md").read_bytes())
    copied.append(dst)

    man = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry_point": "ARTIFACT_GUIDE.md",
        "verification": "python3 analysis/verify_paper_numbers.py (222 checks)",
        "variant": "lite (no checkpoints)" if args.lite else "full",
        "files": {},
    }
    for f in sorted(copied):
        man["files"][str(f.relative_to(REL))] = {
            "sha256": sha256(f), "bytes": f.stat().st_size}
    (REL / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    lines = [f"{sha256(f)}  {f.relative_to(REL)}" for f in sorted(copied)]
    lines.append(f"{sha256(REL / 'MANIFEST.json')}  MANIFEST.json")
    (REL / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    n = len(man["files"])
    total = sum(v["bytes"] for v in man["files"].values())
    unit = 1e6 if args.lite else 1e9
    print(f"staged {n} files, {total/unit:.2f} {'MB' if args.lite else 'GB'} -> {REL}")

    tar = REL.with_suffix(".tar.gz")
    print("compressing (this takes a few minutes) ...")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(REL, arcname=REL.name)
    r = subprocess.run(["shasum", "-a", "256", str(tar)],
                       capture_output=True, text=True)
    print(f"archive: {tar.stat().st_size/(1e6 if args.lite else 1e9):.2f} "
          f"{'MB' if args.lite else 'GB'}")
    print("archive sha256:", r.stdout.split()[0])


if __name__ == "__main__":
    main()
