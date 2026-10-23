#!/usr/bin/env python3
"""Generate the synthetic benchmark with planted interest shifts.

Used for code validation (smoke tests) and for mechanism checks where the
ground-truth noise labels are known.  NOT a substitute for the six public
Web benchmarks in the paper.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import SyntheticConfig, generate_synthetic, preprocess, save_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.pkl", type=Path)
    ap.add_argument("--n-items", default=600, type=int)
    ap.add_argument("--n-groups", default=20, type=int)
    ap.add_argument("--n-train", default=4000, type=int)
    ap.add_argument("--n-valid", default=400, type=int)
    ap.add_argument("--n-test", default=800, type=int)
    ap.add_argument("--shift-prob", default=0.25, type=float)
    ap.add_argument("--seed", default=7, type=int)
    args = ap.parse_args()

    cfg = SyntheticConfig(n_items=args.n_items, n_groups=args.n_groups,
                          n_train=args.n_train, n_valid=args.n_valid,
                          n_test=args.n_test, shift_prob=args.shift_prob,
                          seed=args.seed)
    data = generate_synthetic(cfg)
    data = preprocess(data, min_item_freq=3)  # lighter filter for the tiny vocab
    save_sessions(data, args.out)
    n_shift = int(data.shift_flags.sum()) if data.shift_flags is not None else 0
    print(f"saved {args.out}: items={data.n_items} "
          f"train={len(data.train)} valid={len(data.valid)} "
          f"test={len(data.test)} shift_sessions={n_shift}")


if __name__ == "__main__":
    main()
