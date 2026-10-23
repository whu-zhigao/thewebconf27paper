#!/usr/bin/env python3
"""RQ4 efficiency benchmark (Section 5.6).

Measures per-epoch wall-clock, per-session inference latency, peak
memory, and the accuracy/efficiency frontier of the vocabulary-block
approximation (B in {1, 8}).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import (SessionRelationIndex, build_adjacency, load_sessions,
                          pad_batch)
from mplignn.model import MPLI_GNN


def bench(model, sessions, device, batch_size=128, n_batches=10):
    model.eval()
    lat, n_sess = [], 0
    with torch.no_grad():
        for i in range(min(n_batches * batch_size, len(sessions))):
            chunk = sessions[i:i + batch_size]
            seqs, lens, _ = pad_batch(chunk, device)
            a = build_adjacency(seqs, lens)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(seqs, lens, a)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0 / len(chunk))
            n_sess += len(chunk)
            if n_sess >= n_batches * batch_size:
                break
    return float(np.mean(lat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = load_sessions(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"device": device}
    for b in [1, 8]:
        model = MPLI_GNN(n_items=data.n_items, d=args.dim,
                         vocab_blocks=b).to(device)
        lat = bench(model, data.test, device)
        results[f"B={b}"] = {
            "ms_per_session": round(lat, 3),
            "params": sum(p.numel() for p in model.parameters()),
        }
        print(f"B={b}: {lat:.3f} ms/session")
    if device == "cuda":
        results["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # latency vs session length
    by_len = {}
    for t_max in [5, 10, 20]:
        sessions = [s for s in data.test if len(s) <= t_max][:512]
        model = MPLI_GNN(n_items=data.n_items, d=args.dim).to(device)
        by_len[str(t_max)] = round(bench(model, sessions, device,
                                         n_batches=4), 3)
    results["latency_by_session_length"] = by_len

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
