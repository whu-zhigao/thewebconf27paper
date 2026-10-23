#!/usr/bin/env python3
"""Convert the downloaded raw datasets into SessionData pickles.

Sources (all public, all used by the baselines in the paper):
  diginetica / yoochoose / retailrocket : sessionRec/MTD preprocessed files
      (pickled (instances, targets) with every training prefix)
  nowplaying                            : CCIIPLab/GCE-GNN preprocessed
  gowalla                               : SNAP raw check-ins (top-30k POIs,
      per-user-day sessions, chronological split)

The MTD/GCE-GNN files store augmented prefixes; we reconstruct the maximal
full sessions (a session is prefix+target; keep only prefixes that are not
a prefix of another prefix).
"""
import argparse
import gzip
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mplignn.data import SessionData, save_sessions


def reconstruct_full(obj) -> list:
    """Maximal sessions from (instances, targets) augmented files."""
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], list):
        insts, targets = obj[0], obj[1]
    else:
        insts, targets = obj, None
    # full candidate = inst (+ target when available)
    if targets is not None:
        cands = [tuple(list(s) + [t]) for s, t in zip(insts, targets)]
    else:
        cands = [tuple(s) for s in insts]
    cands.sort()
    out = []
    for i, c in enumerate(cands):
        if i + 1 < len(cands) and cands[i + 1][:len(c)] == c:
            continue  # c is a prefix of a longer session -> not maximal
        if len(c) >= 2:
            out.append(list(c))
    return out


def load_pickle_sessions(path: Path) -> list:
    obj = pickle.loads(path.read_bytes())
    return reconstruct_full(obj)


def remap_and_filter(sessions, min_freq=5):
    from collections import Counter
    freq = Counter(i for s in sessions for i in s)
    keep = {i for i, c in freq.items() if c >= min_freq}
    remap = {i: j + 1 for j, i in enumerate(sorted(keep))}  # 0 reserved for pad
    out, dropped = [], 0
    for s in sessions:
        s2 = [remap[i] for i in s if i in keep]
        if len(s2) >= 2:
            out.append(s2)
        else:
            dropped += 1
    return out, len(remap) + 1, dropped


def convert_mtd_like(name: str, raw_dir: Path, out_dir: Path,
                     valid_frac: float = 0.05):
    train_full = load_pickle_sessions(raw_dir / "train.txt")
    test_full = load_pickle_sessions(raw_dir / "test.txt")
    # items must be remapped consistently over train+test
    all_sessions = train_full + test_full
    freq = defaultdict(int)
    for s in all_sessions:
        for i in set(s):
            freq[i] += 1
    keep = {i for i, c in freq.items() if c >= 5}
    remap = {i: j + 1 for j, i in enumerate(sorted(keep))}

    def _f(seqs):
        out = []
        for s in seqs:
            s2 = [remap[i] for i in s if i in keep]
            if len(s2) >= 2:
                out.append(s2)
        return out

    train_full, test_full = _f(train_full), _f(test_full)
    n_valid = int(len(train_full) * valid_frac)
    valid, train = train_full[-n_valid:], train_full[:-n_valid]
    data = SessionData(train=train, valid=valid, test=test_full,
                       n_items=len(remap) + 1)
    out = out_dir / f"{name}.pkl"
    save_sessions(data, out)
    clicks = sum(len(s) for s in train + valid + test_full)
    print(f"{name}: items={data.n_items} train={len(train)} "
          f"valid={len(valid)} test={len(test_full)} clicks={clicks} "
          f"avg_len={clicks / max(len(train) + len(valid) + len(test_full), 1):.2f}")


def convert_gowalla(raw_dir: Path, out_dir: Path, top_k=30000):
    """SNAP check-ins -> per-user-day sessions, chronological 75/5/20 split."""
    path = raw_dir / "checkins.txt.gz"
    events = []
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            user, ts, loc = p[0], p[1], p[4]
            day = ts.split("T")[0]
            events.append((user, day, ts, loc))
    events.sort(key=lambda e: (e[0], e[2]))
    loc_freq = defaultdict(int)
    for e in events:
        loc_freq[e[3]] += 1
    top = set(sorted(loc_freq, key=lambda l: -loc_freq[l])[:top_k])
    sessions, cur_user_day = [], None
    cur = []
    for user, day, ts, loc in events:
        key = (user, day)
        if key != cur_user_day:
            if len(cur) >= 2:
                sessions.append(cur)
            cur, cur_user_day = [], key
        if loc in top:
            cur.append(loc)
    if len(cur) >= 2:
        sessions.append(cur)
    remap = {l: j + 1 for j, l in enumerate(sorted(top))}
    sessions = [[remap[l] for l in s] for s in sessions]
    sessions = [s for s in sessions if len(s) >= 2]
    n = len(sessions)
    train, valid, test = (sessions[:int(n * 0.75)],
                          sessions[int(n * 0.75):int(n * 0.80)],
                          sessions[int(n * 0.80):])
    data = SessionData(train=train, valid=valid, test=test, n_items=len(remap) + 1)
    save_sessions(data, out_dir / "gowalla.pkl")
    clicks = sum(len(s) for s in sessions)
    print(f"gowalla: items={data.n_items} train={len(train)} "
          f"valid={len(valid)} test={len(test)} clicks={clicks} "
          f"avg_len={clicks / n:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="diginetica,yoochoose,retailrocket,nowplaying,gowalla")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    for name in args.datasets.split(","):
        if name == "gowalla":
            convert_gowalla(args.raw / "gowalla", args.out)
        else:
            convert_mtd_like(name, args.raw / name, args.out)


if __name__ == "__main__":
    main()
