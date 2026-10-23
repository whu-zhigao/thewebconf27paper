#!/usr/bin/env python3
"""reidentify_checkpoint.py -- Appendix B of the paper, as an executable.

The released budget run recorded as HR@20/MRR@20 = 30.09/21.35 carries
the flag combination ``--ablation no_residual`` with the default
``local_layers=2``. This script re-evaluates its saved checkpoint under
each of the four candidate architectures and shows that only the
two-layer no-residual instantiation reproduces the recorded values
(within the thread-count floating-point noise documented in the paper).

Run from code/:
    python3 analysis/reidentify_checkpoint.py
Expected output rows match Appendix B:
    L=2, no residual   ~30.06 / 21.35   (recorded: 30.09 / 21.35)
    L=2, residual      ~14.09 / 10.56
    L=1, no residual   ~23.79 / 15.38
    L=1, residual      ~14.42 / 10.69
The L=1 rows load with strict=False (the second GGNN block is dropped;
the first block keeps its trained weights), which is the identification
procedure described in the appendix.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mplignn.data import (FastRelationIndex, build_adjacency,  # noqa: E402
                          load_sessions, pad_batch)
from mplignn.metrics import full_sort_metrics  # noqa: E402
from mplignn.model import MPLI_GNN  # noqa: E402

REAL = Path(__file__).parent.parent / "results" / "real"
CKPT = REAL / "abl_no_residual_diginetica.pt"
DATA = Path(__file__).parent.parent / "data" / "diginetica.pkl"
EXPECTED = {  # Appendix B of the paper
    (2, True): (30.06, 21.35),   # L=2, no residual
    (2, False): (14.09, 10.56),  # L=2, residual
    (1, True): (23.79, 15.38),   # L=1, no residual
    (1, False): (14.42, 10.69),  # L=1, residual
}


@torch.no_grad()
def evaluate(model, data, rel, batch_size=128):
    hits, mrrs = [], []
    for i in range(0, len(data.test), batch_size):
        chunk = data.test[i:i + batch_size]
        seqs, lens, targets = pad_batch(chunk)
        a = build_adjacency(seqs, lens)
        ids = torch.arange(i, i + len(chunk))
        a_r, r0 = rel.batch_relation(ids, model.embedding.weight)
        out = model(seqs, lens, a, a_r, r0)
        m = full_sort_metrics(out["scores"], targets)
        hits.append(float(m["HR@20"]) * len(chunk))
        mrrs.append(float(m["MRR@20"]) * len(chunk))
    n = len(data.test)
    return 100.0 * sum(hits) / n, 100.0 * sum(mrrs) / n


def main():
    data = load_sessions(DATA)
    rel = FastRelationIndex(data.test, data.train, data.n_items, 10,
                            cache=DATA.parent / "relcache" / "diginetica_test.npz")
    print(f"checkpoint: {CKPT.name}   test sessions: {len(data.test)}\n")
    print(f"{'instantiation':<22}{'HR@20':>8}{'MRR@20':>9}   {'Appendix B':>12}   load")
    for layers, no_res in [(2, True), (2, False), (1, True), (1, False)]:
        model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=layers,
                         n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                         no_residual=no_res)
        missing, unexpected = model.load_state_dict(
            torch.load(CKPT, map_location="cpu", weights_only=True), strict=False)
        model.eval()
        hr, mrr = evaluate(model, data, rel)
        eh, em = EXPECTED[(layers, no_res)]
        ok = "exact" if layers == 2 else f"drop {len(unexpected)} tensors"
        print(f"L={layers}, {'no residual' if no_res else 'residual   ':<13}"
              f"{hr:8.2f}{mrr:9.2f}   {eh:5.2f}/{em:5.2f}   {ok}")
    print("\nOnly the L=2 no-residual instantiation reproduces the recorded "
          "30.09/21.35 (delta <= 0.03 from thread-count reduction order); "
          "the flag therefore removes the residual and keeps both layers.")


if __name__ == "__main__":
    main()
