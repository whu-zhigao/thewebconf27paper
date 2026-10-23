#!/usr/bin/env python3
"""Build a self-contained GPU upload package (mplignn_gpu_package.tar.gz).

Contents:
  code/    mplignn library, scripts, run_gpu.py, NOTES, tests
  data/    the three full-protocol dataset pickles
  data/relcache/  precomputed relation indices for those datasets
           (saves ~40 min of index building on the GPU node)
  GPU_RUNBOOK.md  step-by-step instructions

Re-run any time; the package is deterministic apart from the tar timestamp.
"""
import shutil
import tarfile
from pathlib import Path

P = Path(__file__).parent
STAGE = P / "_gpu_stage"
NAME = "mplignn_gpu_package.tar.gz"

CODE = ["mplignn", "scripts", "run_gpu.py", "run_all_gpu.py",
        "NOTES.md", "README.md"]
DATA = ["data/diginetica.pkl", "data/gowalla.pkl", "data/yoochoose.pkl",
        "data/yoochoose_quarter.pkl", "data/nowplaying.pkl",
        "data/retailrocket.pkl", "data/longchain.pkl"]
CACHE = [f"data/relcache/{d}_{s}.npz"
         for d in ("diginetica", "gowalla", "yoochoose",
                   "yoochoose_quarter", "nowplaying", "retailrocket")
         for s in ("train", "valid", "test")]
SKIP = {"__pycache__", ".ipynb_checkpoints"}


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "code").mkdir(parents=True)
    n, missing = 0, []
    for top in CODE:
        src = P / top
        if not src.exists():
            missing.append(top)
            continue
        if src.is_dir():
            shutil.copytree(src, STAGE / "code" / top,
                            ignore=shutil.ignore_patterns(*SKIP))
            n += sum(1 for _ in (STAGE / "code" / top).rglob("*"))
        else:
            shutil.copy(src, STAGE / "code" / top)
            n += 1
    for rel in DATA + CACHE:
        src = P / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = STAGE / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        n += 1
    shutil.copy(Path(__file__).parent / "GPU_RUNBOOK.md",
                STAGE / "code" / "GPU_RUNBOOK.md")
    (STAGE / "code" / "results" / "real").mkdir(parents=True, exist_ok=True)

    out = P / NAME
    with tarfile.open(out, "w:gz") as tf:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                tf.add(f, arcname=str(f.relative_to(STAGE)))
    print(f"{out.name}: {out.stat().st_size/1e6:.1f} MB, {n} entries copied")
    if missing:
        print("MISSING (package incomplete):", missing)
    else:
        print("package complete -- scp it to the GPU server (see GPU_RUNBOOK.md)")


if __name__ == "__main__":
    main()
