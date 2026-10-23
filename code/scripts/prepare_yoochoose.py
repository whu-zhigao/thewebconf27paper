#!/usr/bin/env python3
"""Build the Yoochoose 1/4 subsample from the raw RecSys-2015 clickstream.

Protocol (validated against the published SR-GNN statistics):
  * clicks only, sessions ordered by timestamp,
  * the standard 1/64 file from the baseline repositories contains
    118,479 full training sessions, whose prefix augmentation gives
    exactly the published 369,859 training instances;
  * we therefore reuse the 1/64 test sessions for the 1/4 file (identical
    test set, as in the literature) and take the most recent quarter of
    the chronological training pool from the raw stream,
  * items are filtered jointly on the 1/4 train and the shared test set.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mplignn.data import SessionData, load_sessions, save_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", type=Path,
                    default=Path("data/raw/yoochoose/yoochoose-clicks.dat"))
    ap.add_argument("--base", type=Path, default=Path("data/yoochoose.pkl"),
                    help="existing 1/64 SessionData with the shared test set")
    ap.add_argument("--out", type=Path,
                    default=Path("data/yoochoose_quarter.pkl"))
    args = ap.parse_args()

    base = load_sessions(args.base)
    print(f"base 1/64: train={len(base.train):,} test={len(base.test):,} "
          f"items={base.n_items:,}")

    print("reading raw click stream (~33M rows)...", flush=True)
    df = pd.read_csv(args.clicks, usecols=[0, 1, 2],
                     names=["session", "time", "item"],
                     dtype={"session": "int64", "item": "int64"})
    df["time"] = pd.to_datetime(df["time"])
    first = df.groupby("session")["time"].min().sort_values()
    # chronological training pool = all but the most recent 4 days of
    # sessions (the standard test window covers those days)
    cutoff = first.iloc[-55_000:].min()
    pool_ids = first.index[first < cutoff]
    n_quarter = len(pool_ids) // 4
    sub_ids = set(pool_ids[-n_quarter:])
    print(f"pool={len(pool_ids):,} cutoff={cutoff} "
          f"quarter={len(sub_ids):,} sessions")

    keep_rows = df["session"].isin(sub_ids)
    sub = df[keep_rows].sort_values(["session", "time"], kind="mergesort")
    train_raw = {sid: g["item"].tolist()
                 for sid, g in sub.groupby("session", sort=False)}

    # test window: the last two (partially observed) days of the stream,
    # which after train-frequency filtering yields ~11K sessions, the same
    # scale as the 1/64 benchmark's test set
    test_ids = set(first.index[first >= pd.Timestamp("2014-09-29", tz="UTC")])
    test_rows = df[df["session"].isin(test_ids)].sort_values(
        ["session", "time"], kind="mergesort")
    test_raw = {sid: g["item"].tolist()
                for sid, g in test_rows.groupby("session", sort=False)}
    print(f"raw test window sessions: {len(test_raw):,}")

    # item space from TRAIN frequency only (standard protocol: test items
    # unseen in training are dropped from test sessions)
    freq = Counter(i for s in train_raw.values() for i in s)
    keep = {i for i, c in freq.items() if c >= 5}
    remap = {i: j + 1 for j, i in enumerate(sorted(keep))}
    train = [[remap[i] for i in s if i in keep] for s in train_raw.values()]
    train = [s for s in train if len(s) >= 2]
    test = [[remap[i] for i in s if i in keep] for s in test_raw.values()]
    test = [s for s in test if len(s) >= 2]

    n_valid = int(len(train) * 0.05)
    valid, train = train[-n_valid:], train[:-n_valid]
    data = SessionData(train=train, valid=valid, test=test,
                       n_items=len(keep) + 1)
    save_sessions(data, args.out)
    clicks = sum(len(s) for s in train + valid + test)
    print(f"wrote {args.out}: items={data.n_items:,} "
          f"train={len(train):,} valid={len(valid):,} test={len(test):,} "
          f"clicks={clicks:,}")


if __name__ == "__main__":
    main()
