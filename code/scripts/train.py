#!/usr/bin/env python3
"""Train MPLI-GNN / SR-GNN-lite / Item-KNN on a real dataset (RQ1/RQ2)."""
import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import (FastRelationIndex, SessionData, build_adjacency,
                          load_sessions, pad_batch)
from mplignn.metrics import full_sort_metrics
from mplignn.model import MPLI_GNN, SRGNNLite
from mplignn.baselines import BASELINES


# ---------------------------------------------------------------------------
# Item-KNN baseline (co-occurrence transition scoring, no training)
# ---------------------------------------------------------------------------
class ItemKNN:
    def __init__(self, data: SessionData):
        self.n_items = data.n_items
        # sparse transition store: item -> (neighbor_ids, counts)
        acc = defaultdict(lambda: defaultdict(float))
        for s in data.train:
            for a_i, b_i in zip(s[:-1], s[1:]):
                acc[a_i][b_i] += 1.0
        self.trans = {
            i: (np.fromiter(d.keys(), dtype=np.int64),
                np.fromiter(d.values(), dtype=np.float32))
            for i, d in acc.items()}

    @torch.no_grad()
    def evaluate(self, sessions, device="cpu", batch_size=128, **kw):
        agg, n = defaultdict(float), 0
        bagg, n_sess = defaultdict(float), 0
        top10_global, top20_global = set(), set()
        for i in range(0, len(sessions), batch_size):
            chunk = sessions[i:i + batch_size]
            seqs, lens, targets = pad_batch(chunk, device)
            scores = np.zeros((len(chunk), self.n_items), dtype=np.float32)
            for b, s in enumerate(chunk):
                row = np.zeros(self.n_items, dtype=np.float32)
                for it in s[:-1]:
                    if it in self.trans:
                        nb, cnt = self.trans[it]
                        np.add.at(row, nb, cnt)
                scores[b] = row
            for k, v in full_sort_metrics(torch.from_numpy(scores).to(targets.device),
                                          targets).items():
                agg[k] += v
                if not k.startswith("Coverage"):
                    bagg[k] += v * len(chunk)
            top = torch.from_numpy(scores).to(targets.device).topk(20, dim=-1).indices
            top20_global.update(top.reshape(-1).tolist())
            top10_global.update(top[:, :10].reshape(-1).tolist())
            n += 1
            n_sess += len(chunk)
        res = {k: v / n for k, v in agg.items()}
        for k, v in bagg.items():
            res[k + "_fixed"] = v / n_sess
        res["Coverage@10_global"] = len(top10_global) / self.n_items
        res["Coverage@20_global"] = len(top20_global) / self.n_items
        return res


# ---------------------------------------------------------------------------
# Shared evaluation with cached relation inputs
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, sessions, rel: FastRelationIndex, device,
             batch_size=128, noise_ratio=0.0, rng=None):
    model.eval()
    agg, n = defaultdict(float), 0
    bagg, n_sess = defaultdict(float), 0
    top10_global, top20_global = set(), set()
    for i in range(0, len(sessions), batch_size):
        chunk = sessions[i:i + batch_size]
        if noise_ratio > 0:
            rng = rng or np.random.default_rng(0)
            chunk = [list(s) for s in chunk]
            for s in chunk:
                n_noisy = int(round(noise_ratio * len(s)))
                if n_noisy:
                    for pos in rng.choice(len(s), size=n_noisy, replace=False):
                        s[pos] = int(rng.integers(1, model.n_items))
        ids = torch.arange(i, i + len(chunk))
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        seqs, lens, targets = pad_batch(chunk, device)
        a = build_adjacency(seqs, lens)
        out = model(seqs, lens, a, a_r, r0)
        for k, v in full_sort_metrics(out["scores"], targets).items():
            agg[k] += v
            if not k.startswith("Coverage"):
                bagg[k] += v * len(chunk)
        top = out["scores"].topk(20, dim=-1).indices
        top20_global.update(top.reshape(-1).tolist())
        top10_global.update(top[:, :10].reshape(-1).tolist())
        n += 1
        n_sess += len(chunk)
    res = {k: v / n for k, v in agg.items()}
    for k, v in bagg.items():
        res[k + "_fixed"] = v / n_sess
    n_items = getattr(model, "n_items", out["scores"].shape[-1])
    res["Coverage@10_global"] = len(top10_global) / n_items
    res["Coverage@20_global"] = len(top20_global) / n_items
    return res



def _peak_rss_mb() -> float:
    """Peak RSS in MB, portable across macOS (bytes) and Linux (KB)."""
    import sys as _sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if _sys.platform == "darwin" else raw / 1e3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model",
                    choices=["mplignn", "srgnn", "itemknn"] + list(BASELINES),
                    default="mplignn")
    ap.add_argument("--ablation", default="",
                    help="no_residual,no_global,no_anchor,softmax_attn,"
                         "no_denoise,avg_intent")
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--local-layers", default=2, type=int)
    ap.add_argument("--global-blocks", default=2, type=int)
    ap.add_argument("--vocab-blocks", default=1, type=int)
    ap.add_argument("--heads", default=2, type=int)
    ap.add_argument("--k-r", default=10, type=int)
    ap.add_argument("--dropout", default=0.1, type=float)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight-decay", default=1e-5, type=float)
    ap.add_argument("--epochs", default=15, type=int)
    ap.add_argument("--patience", default=3, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--max-batches", default=0, type=int,
                    help="cap batches per epoch (0 = full pass)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--depth", default=1, type=int,
                    help="local depth for srgnn/gcegnn (audit)")
    ap.add_argument("--no-global", action="store_true",
                    help="disable the global stream (gcegnn audit)")
    ap.add_argument("--rel-control", default="none",
                    choices=["none", "rand", "shuffle", "frozen"],
                    help="revision M3 training-time memory control: "
                         "rand=uniform random training sessions, "
                         "shuffle=misaligned retrieved sets, "
                         "frozen=stop gradient through the memory side")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.rel_control != "none" and args.model != "mplignn":
        ap.error("--rel-control applies to --model mplignn only")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = args.device
    data: SessionData = load_sessions(args.data)
    stem = args.data.stem

    if args.model == "itemknn":
        t0 = time.time()
        knn = ItemKNN(data)
        test = knn.evaluate(data.test)
        result = {"model": "itemknn", "test": test,
                  "wall_seconds": time.time() - t0}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print("TEST", json.dumps(test, indent=2))
        return

    flags = {f.strip(): True for f in args.ablation.split(",") if f.strip()}
    if args.model in BASELINES:
        kw = {"dropout": args.dropout}
        if args.model == "gcegnn":
            kw.update(depth=args.depth,
                      use_global=not args.no_global)
        model = BASELINES[args.model](n_items=data.n_items, d=args.dim,
                                      **kw).to(device)
    else:
        model = (SRGNNLite(n_items=data.n_items, d=args.dim,
                       dropout=args.dropout,
                       depth=args.depth) if args.model == "srgnn"
             else MPLI_GNN(n_items=data.n_items, d=args.dim,
                           local_layers=args.local_layers,
                           n_heads=args.heads,
                           global_blocks=args.global_blocks,
                           vocab_blocks=args.vocab_blocks,
                           k_r=args.k_r, dropout=args.dropout,
                           **flags)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    cache_dir = args.data.parent / "relcache"
    cache_dir.mkdir(exist_ok=True)

    def _cache(split: str) -> Path:
        # k_r-specific cache name (revision fix: the old name silently
        # reused the k=10 index for any --k-r).  k=10 keeps the legacy
        # filename so existing artifacts stay valid.
        p = cache_dir / f"{stem}_{split}_k{args.k_r}.npz"
        if p.exists():
            return p
        if args.k_r == 10:
            legacy = cache_dir / f"{stem}_{split}.npz"
            if legacy.exists():
                return legacy
        return p

    rel_train = FastRelationIndex(data.train, data.train, data.n_items,
                                  args.k_r, cache=_cache("train"),
                                  keep_corpus=args.rel_control == "rand")
    rel_valid = FastRelationIndex(data.valid, data.train, data.n_items,
                                  args.k_r, cache=_cache("valid"))
    rel_test = FastRelationIndex(data.test, data.train, data.n_items,
                                 args.k_r, cache=_cache("test"))

    best_mrr, best_state, best_epoch, idle = -1.0, None, -1, 0
    history = []
    epoch_seconds = []
    t0 = time.time()
    n_train = len(data.train)
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        perm = np.random.permutation(n_train)
        limit = len(perm)
        if args.max_batches > 0:
            limit = min(limit, args.max_batches * args.batch_size)
        total_loss, n_batches = 0.0, 0
        for i in range(0, limit, args.batch_size):
            ids = torch.from_numpy(perm[i:i + args.batch_size])
            chunk = [data.train[j] for j in ids.tolist()]
            seqs, lens, targets = pad_batch(chunk, device)
            a = build_adjacency(seqs, lens)
            ctrl = args.rel_control if args.model == "mplignn" else "none"
            a_r, r0 = rel_train.batch_relation(
                ids, model.embedding.weight,
                control="none" if ctrl == "frozen" else ctrl,
                detach_memory=ctrl == "frozen")
            out = model(seqs, lens, a, a_r, r0)
            loss = F.cross_entropy(out["scores"], targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        vm = evaluate(model, data.valid, rel_valid, device, args.batch_size)
        epoch_seconds.append(round(time.time() - t_epoch, 2))
        history.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1),
                        "valid": vm})
        print(f"epoch {epoch:3d} loss {total_loss / max(n_batches, 1):.4f} "
              f"val MRR@20 {vm['MRR@20']:.4f} HR@20 {vm['HR@20']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if vm["MRR@20"] > best_mrr:
            best_mrr, best_epoch, idle = vm["MRR@20"], epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            idle += 1
            if idle >= args.patience:
                break

    model.load_state_dict(best_state)
    import time as _t
    lat0 = _t.perf_counter()
    test = evaluate(model, data.test, rel_test, device, args.batch_size)
    test_seconds = _t.perf_counter() - lat0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = args.out.with_suffix(".pt")
    torch.save(best_state, ckpt)
    result = {"model": args.model, "ablation": flags, "seed": args.seed,
              "rel_control": args.rel_control, "k_r": args.k_r,
              "dataset": stem, "dim": args.dim, "epochs": args.epochs,
              "max_batches": args.max_batches,
              "best_epoch": best_epoch, "wall_seconds": time.time() - t0,
              "valid": history[-1]["valid"] if history else {},
              "test": test, "history": history,
              "epoch_seconds": epoch_seconds,
              "sec_per_epoch": round(sum(epoch_seconds) / len(epoch_seconds), 2)
                  if epoch_seconds else None,
              "ms_per_session": round(test_seconds * 1000 / max(len(data.test), 1), 2),
              "peak_rss_mb": round(_peak_rss_mb(), 1),
              "total_hours": round((time.time() - t0) / 3600, 3),
              "params": sum(p.numel() for p in model.parameters())}
    args.out.write_text(json.dumps(result, indent=2))
    print("TEST", json.dumps(test, indent=2))
    print(f"wrote {args.out} + {ckpt} (best epoch {best_epoch})")


if __name__ == "__main__":
    main()
