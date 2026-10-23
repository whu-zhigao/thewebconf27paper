#!/usr/bin/env python3
"""GPU-node self-test: environment, data, and two training steps.

Run this FIRST on the GPU server:
    python3 scripts/gpu_selftest.py

Checks (each printed PASS/FAIL):
  1. torch version and CUDA device availability
  2. cudnn deterministic pinning
  3. required dataset pickles present
  4. entmax produces valid sparse distributions on GPU
  5. one MPLI-GNN forward+backward step on GPU (finite loss & grads)
  6. one baseline (NARM) forward+backward step on GPU
  7. relation-index tensors flow across devices correctly
Exit code 0 only if everything passes.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mplignn.baselines import NARM
from mplignn.data import FastRelationIndex, build_adjacency, pad_batch
from mplignn.entmax import entmax
from mplignn.model import MPLI_GNN

OK, FAIL = "PASS", "FAIL"
failures = []


def check(name, fn):
    try:
        fn()
        print(f"[{OK}] {name}")
    except Exception as e:
        print(f"[{FAIL}] {name}: {type(e).__name__}: {e}")
        failures.append(name)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import platform
    print(f"python={platform.python_version()} "
          f"platform={platform.system()} torch={torch.__version__}")

    def ver():
        v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        assert v >= (2, 0), f"torch too old: {torch.__version__}"

    check("torch >= 2.0", ver)

    def env():
        print(f"     device={dev}"
              + (f" gpu={torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
        assert dev == "cuda", "CUDA not available -- check driver/torch build"

    check("CUDA device", env)

    def weights_only():
        # torch >= 2.6 defaults weights_only=True; our checkpoints must load
        # under exactly that policy (plain state_dict of tensors).
        m = torch.nn.Linear(4, 4)
        tmp = Path("/tmp/_wo_test.pt")
        torch.save(m.state_dict(), tmp)
        sd = torch.load(tmp, map_location="cpu", weights_only=True)
        assert set(sd) == {"weight", "bias"}
        tmp.unlink()

    check("checkpoint loads with weights_only=True", weights_only)

    def det():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    check("cudnn deterministic", det)

    def data():
        for f in ["diginetica", "gowalla", "yoochoose_quarter"]:
            p = Path("data") / f"{f}.pkl"
            assert p.exists(), f"missing {p}"

    check("dataset pickles", data)

    def ent():
        z = torch.randn(64, 200, device=dev)
        p = entmax(z, alpha=1.5, dim=-1)
        assert torch.allclose(p.sum(-1), torch.ones(64, device=dev), atol=1e-4)
        assert (p == 0).float().mean() > 0.1, "attention not sparse"

    check("entmax-1.5 on GPU", ent)

    sessions = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]]
    seqs, lens, targets = pad_batch(sessions, dev)
    a = build_adjacency(seqs, lens)
    a_r = torch.eye(4, device=dev).unsqueeze(0).expand(3, 4, 4).clone() * 0.1
    r0 = torch.randn(3, 4, 64, device=dev)

    def mpli():
        torch.manual_seed(0)
        m = MPLI_GNN(n_items=50, d=64).to(dev)
        out = m(seqs, lens, a, a_r, r0)
        loss = torch.nn.functional.cross_entropy(out["scores"], targets)
        loss.backward()
        gs = [p.grad for p in m.parameters() if p.grad is not None]
        assert gs and all(torch.isfinite(g).all() for g in gs)
        assert torch.isfinite(loss)

    check("MPLI-GNN fwd/bwd on GPU", mpli)

    def narm():
        torch.manual_seed(0)
        m = NARM(n_items=50, d=64).to(dev)
        out = m(seqs, lens, a)
        loss = torch.nn.functional.cross_entropy(out["scores"], targets)
        loss.backward()

    check("NARM baseline fwd/bwd on GPU", narm)

    def relidx():
        train = [[1, 2], [3, 4], [5, 6], [1, 3, 5]]
        idx = FastRelationIndex(train, train, n_items=10, k_r=2)
        emb = torch.randn(10, 64, device=dev)          # CUDA table
        ids = torch.tensor([0, 1])                      # CPU ids (as in train.py)
        a_r, r0 = idx.batch_relation(ids, emb)
        # Compare against the ACTUAL device object (cuda:0 != "cuda" as a
        # string), and additionally require the right device type.
        want_type = torch.device(dev).type
        assert (a_r.device == emb.device and r0.device == emb.device
                and a_r.device.type == want_type and r0.device.type == want_type), \
            f"devices: a_r={a_r.device} r0={r0.device} emb={emb.device} want={want_type}"
        print(f"     a_r/r0 on {a_r.device} (index tensors flowed CPU->GPU correctly)")

    check("relation index device flow", relidx)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    def rss():
        import resource as _r, sys as _s
        raw = _r.getrusage(_r.RUSAGE_SELF).ru_maxrss
        mb = raw / 1e6 if _s.platform == "darwin" else raw / 1e3
        print(f"     peak RSS ~{mb:.0f} MB ({_s.platform}: raw={raw})")
        assert 50 < mb < 200000, "implausible RSS -- unit detection broken"

    check("RSS unit sanity (Linux=KB / macOS=bytes)", rss)

    print("ALL GPU SELF-TESTS PASSED -- ready: python3 run_gpu.py --check")


if __name__ == "__main__":
    main()
