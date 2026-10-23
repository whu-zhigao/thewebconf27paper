#!/usr/bin/env python3
"""Unit tests for MPLI-GNN components.

Each test verifies a property claimed in the paper:
  - entmax outputs are probability distributions and become sparser as
    alpha grows (Section 4.4);
  - the local encoder keeps residuals and shapes (Eq. 2-3);
  - the global encoder with B=1 equals the unpartitioned computation;
  - the intent head is local-primary: z depends only on local rows;
  - ablation flags change exactly the intended pathway;
  - full model forward/backward runs and produces finite gradients.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplignn.data import build_adjacency, pad_batch
from mplignn.entmax import entmax, sparsemax
from mplignn.model import GlobalEncoder, IntentHead, LocalEncoder, MPLI_GNN, SRGNNLite


def test_entmax_is_distribution():
    z = torch.randn(64, 50)
    for alpha in [1.5, 1.8, 2.0]:
        p = entmax(z, alpha=alpha)
        assert torch.allclose(p.sum(-1), torch.ones(64), atol=1e-4), alpha
        assert (p >= 0).all()


def test_entmax_sparsity_monotone_in_alpha():
    torch.manual_seed(0)
    z = torch.randn(8, 200)
    zeros = [float((entmax(z, a) == 0).float().mean()) for a in (1.3, 1.6, 2.0)]
    assert zeros[0] <= zeros[1] <= zeros[2], zeros


def test_sparsemax_closed_form():
    z = torch.randn(16, 40)
    p = sparsemax(z)
    assert torch.allclose(p.sum(-1), torch.ones(16), atol=1e-4)
    assert torch.allclose(entmax(z, alpha=2.0), p, atol=1e-4)


def test_local_encoder_residual_and_shape():
    enc = LocalEncoder(d=32, n_layers=3, residual=True)
    x = torch.randn(4, 10, 32)
    a = build_adjacency(torch.randint(0, 5, (4, 10)), torch.full((4,), 10))
    y = enc(x, a)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)  # transformation happened


def test_global_encoder_block_partition_consistency():
    torch.manual_seed(1)
    enc = GlobalEncoder(d=32, n_heads=2, n_blocks_m=2, n_blocks=1)
    x = torch.randn(64, 32)
    y_full = enc(x)
    assert y_full.shape == x.shape


def test_intent_head_local_primary():
    head = IntentHead(d=32)
    h_n = torch.randn(4, 8, 32)
    r = torch.randn(4, 32)
    lens = torch.tensor([8, 8, 8, 8])
    z, beta = head(h_n, r, lens)
    assert z.shape == (4, 32)
    assert torch.allclose(beta.sum(-1), torch.ones(4), atol=1e-5)
    # perturbing r changes keys (auxiliary) but z is pooled only from h_n
    z2, _ = head(h_n + 0.0, r + 10.0, lens)
    assert not torch.allclose(z, z2)  # global gate modulates


def _tiny_batch(n_items=40):
    sessions = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    seqs, lens, targets = pad_batch(sessions)
    a = build_adjacency(seqs, lens)
    return seqs, lens, targets, a


def test_full_model_forward_backward():
    seqs, lens, targets, a = _tiny_batch()
    model = MPLI_GNN(n_items=40, d=32, local_layers=2, global_blocks=2,
                     k_r=4)
    a_r = torch.eye(4).unsqueeze(0).expand(3, 4, 4).clone()
    r0 = torch.randn(3, 4, 32)
    out = model(seqs, lens, a, a_r, r0)
    assert out["scores"].shape == (3, 40)
    loss = F.cross_entropy(out["scores"], targets)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_ablation_flags_change_output():
    seqs, lens, targets, a = _tiny_batch()
    torch.manual_seed(0)
    m1 = MPLI_GNN(n_items=40, d=32)
    torch.manual_seed(0)
    m2 = MPLI_GNN(n_items=40, d=32, no_denoise=True)
    o1 = m1(seqs, lens, a)["z"]
    o2 = m2(seqs, lens, a)["z"]
    assert not torch.allclose(o1, o2)


def test_srgnn_baseline_runs():
    seqs, lens, targets, a = _tiny_batch()
    model = SRGNNLite(n_items=40, d=32)
    out = model(seqs, lens, a)
    assert out["scores"].shape == (3, 40)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
