#!/usr/bin/env python3
# M8 eval replay: re-evaluate released checkpoints under the corrected
# session-weighted metric and global-coverage counting, WITHOUT touching
# the original run records (all 222 released assertions stay valid).
#
# For every results/real/*.json that has a sibling .pt checkpoint and a
# reconstructable mplignn configuration, this script rebuilds the model,
# loads the checkpoint, re-runs test evaluation with the dual-metric
# evaluate() from scripts/train.py, and writes <stem>_replay.json:
#
#   {"source": "<stem>.json", "config": {...}, "test_fixed": {...}}
#
# Configuration reconstruction: model/ablation/dim/seed come from the JSON;
# local_layers is derived from the D-cell naming convention (D1* -> 1,
# else 2); remaining hyperparameters use the released defaults
# (heads=2, global_blocks=2, vocab_blocks=1, k_r=10).
#
# Usage:
#   python3 analysis/replay_eval.py                       # all mplignn cells
#   python3 analysis/replay_eval.py --pattern 'D0R1*'     # subset
#   python3 analysis/replay_eval.py --limit 5 --force
import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

# import the dual-metric evaluate() without running train.main()
_ARGV = sys.argv                      # preserve OUR CLI flags across the
                                     # train.py import shield below
spec = importlib.util.spec_from_file_location("trainmod",
                                              CODE / "scripts" / "train.py")
trainmod = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]           # shield train.py from our CLI flags
spec.loader.exec_module(trainmod)
sys.argv = _ARGV                    # restore before main() re-parses

from mplignn.data import FastRelationIndex, load_sessions
from mplignn.model import MPLI_GNN


def local_layers_from_name(stem, rec):
    if stem.startswith("D1"):
        return 1
    if stem.startswith("seed_mplignn") and (rec.get("ablation") or {}).get(
            "no_residual"):
        return 2                      # v1 no_residual keeps two layers
    return rec.get("local_layers", 2)


def replay_one(path, device, data_dir):
    stem = Path(path).stem
    rec = json.loads(Path(path).read_text())
    if rec.get("model") != "mplignn":
        return None
    ckpt = Path(path).with_suffix(".pt")
    if not ckpt.exists():
        return None
    data = load_sessions(Path(data_dir) / f"{rec['dataset']}.pkl")
    flags = rec.get("ablation") or {}
    model = MPLI_GNN(n_items=data.n_items, d=rec.get("dim", 64),
                     local_layers=local_layers_from_name(stem, rec),
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=bool(flags.get("no_residual")),
                     no_global=bool(flags.get("no_global")),
                     no_anchor=bool(flags.get("no_anchor")),
                     softmax_attn=bool(flags.get("softmax_attn")),
                     no_denoise=bool(flags.get("no_denoise")),
                     avg_intent=bool(flags.get("avg_intent"))).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    relcache = Path(data_dir) / "relcache"
    rel_test = FastRelationIndex(data.test, data.train, data.n_items, 10,
                                  cache=relcache / f"{rec['dataset']}_test.npz")
    test = trainmod.evaluate(model, data.test, rel_test, device)
    fixed = {k: v for k, v in test.items() if k.endswith("_fixed")
             or k.endswith("_global")}
    legacy_drift = {k: round(100 * (test[k + "_fixed"] - test[k]), 4)
                    for k in ("HR@20", "MRR@20")
                    if k + "_fixed" in test}
    return {"source": Path(path).name,
            "config": {"model": "mplignn", "dim": rec.get("dim", 64),
                       "local_layers": local_layers_from_name(stem, rec),
                       "ablation": flags, "seed": rec.get("seed")},
            "test_fixed": fixed, "legacy_minus_fixed_points": legacy_drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*",
                    help="glob over results/real filenames (default all)")
    ap.add_argument("--results-dir", type=Path,
                    default=CODE / "results" / "real",
                    help="directory holding the run records")
    ap.add_argument("--data-dir", type=Path, default=CODE / "data",
                    help="directory holding the dataset pickles + relcache")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing *_replay.json side-cars")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    files = sorted(glob.glob(str(args.results_dir / f"{args.pattern}.json")))
    files = [f for f in files if not f.endswith("_replay.json")]
    if args.limit:
        files = files[:args.limit]
    n_ok = n_skip = 0
    for f in files:
        out = Path(f).with_name(Path(f).stem + "_replay.json")
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            res = replay_one(f, args.device, args.data_dir)
        except Exception as e:                       # noqa: BLE001
            print(f"ERR  {Path(f).name}: {type(e).__name__}: {e}")
            continue
        if res is None:
            n_skip += 1
            continue
        out.write_text(json.dumps(res, indent=2))
        n_ok += 1
        d = res["legacy_minus_fixed_points"]
        print(f"OK   {Path(f).name}  drift(pts)={d}")
    print(f"replayed={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
