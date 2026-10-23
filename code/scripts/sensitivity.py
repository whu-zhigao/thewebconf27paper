#!/usr/bin/env python3
"""RQ3 sensitivity sweep (Section 5.5): L, M_b, h, d, and alpha policy.

Runs train.py in-process for each configuration and writes a JSON of
curves ready for pgfplots.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_one(data, tag, out_dir, dim=64, local_layers=2, global_blocks=2,
            heads=2, ablation="", epochs=10, device="cpu", lr=5e-4,
            patience=4, max_batches=300):
    import subprocess
    out = Path(out_dir) / f"sens_{tag}.json"
    if out.exists():                                  # resumable
        return json.loads(out.read_text())["test"]
    cmd = [sys.executable, str(Path(__file__).parent / "train.py"),
           "--data", str(data), "--model", "mplignn",
           "--dim", str(dim), "--local-layers", str(local_layers),
           "--global-blocks", str(global_blocks), "--heads", str(heads),
           "--epochs", str(epochs), "--patience", str(patience),
           "--max-batches", str(max_batches), "--lr", str(lr),
           "--device", device, "--out", str(out)]
    if ablation:
        cmd += ["--ablation", ablation]
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", default=6, type=int)
    ap.add_argument("--device", default="cpu")
    sweep_dir = ap.parse_args().out.parent / f"sweep_{ap.parse_args().data.stem}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    args = ap.parse_args()

    curves = {"local_depth_L": {}, "global_blocks_M": {}, "heads_h": {},
              "dim_d": {}, "alpha_policy": {}}
    for l in [1, 2, 3, 4]:
        curves["local_depth_L"][str(l)] = run_one(
            args.data, f"L{l}", sweep_dir,
            local_layers=l, epochs=args.epochs, device=args.device)
    for m in [1, 2, 3, 4]:
        curves["global_blocks_M"][str(m)] = run_one(
            args.data, f"M{m}", sweep_dir,
            global_blocks=m, epochs=args.epochs, device=args.device)
    for h in [1, 2, 4, 8]:
        curves["heads_h"][str(h)] = run_one(
            args.data, f"h{h}", sweep_dir,
            heads=h, epochs=args.epochs, device=args.device)
    for d in [32, 64, 128]:
        curves["dim_d"][str(d)] = run_one(
            args.data, f"d{d}", sweep_dir,
            dim=d, epochs=args.epochs, device=args.device)
    curves["alpha_policy"]["entmax"] = run_one(
        args.data, "entmax", sweep_dir, epochs=args.epochs, device=args.device)
    curves["alpha_policy"]["softmax"] = run_one(
        args.data, "softmax", sweep_dir,
        ablation="softmax_attn", epochs=args.epochs)

    args.out.write_text(json.dumps(curves, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
