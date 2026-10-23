# MPLI-GNN Reference Implementation and Experiment Suite

This directory contains the complete, runnable implementation of the
paper *A Factorial Anatomy of Global Memory in Session-Based
Recommendation* together with the drivers for every experiment in its
evaluation section.

## Layout

```
code/
├── mplignn/               # library
│   ├── entmax.py          # alpha-EntMax via bisection + sparsemax closed form
│   ├── data.py            # sessions, local graphs, synthetic shifts, retrieval
│   ├── model.py           # MPLI-GNN + SR-GNN-lite baseline + ablation flags
│   ├── baselines.py       # the twelve re-implemented neural baselines (NOTES.md)
│   └── metrics.py         # HR@k / MRR@k / NDCG@k / Coverage@k, full sort
├── scripts/               # 20 experiment drivers
│   ├── train.py           # single-run entry point (every model + ablation flag)
│   ├── prepare_datasets.py, prepare_yoochoose.py  # split rebuild from public sources
│   ├── pilot_study.py     # motivating measurements (P1-P3)
│   ├── theory_validation.py, calibration_transfer.py  # gate-threshold theory
│   ├── depth_gate_audit.py         # cross-architecture gate audit
│   ├── make_synthetic.py, discriminating_test.py    # discriminating corpus
│   ├── sensitivity.py, noise_injection.py, stratify_shift.py, stratify_many.py
│   ├── efficiency.py, diagnostics_readout.py, case_study.py
│   └── gpu_selftest.py, ingest_gpu_results.py, summarize_real.py,
│       update_paper_results.py
├── analysis/              # table generation + verification (no training)
│   ├── verify_paper_numbers.py   # one-command re-check of 222 paper numbers
│   ├── make_tables.py            # regenerates the empirical LaTeX tables
│   ├── run_cells.py              # enumerates + re-runs factorial cells
│   ├── probe_dead_paths.py       # dead-module probe: bypass + gradient checks
│   ├── theory_fullprec.py        # unrounded λ/margin/τ for Table 4 raw L*
│   ├── reidentify_checkpoint.py  # checkpoint re-identification (Appendix B)
│   └── analysis_seeds.py         # seed-level statistics
├── tests/test_model.py    # unit tests of the module properties claimed in §4
├── results/real/          # one JSON run record per run + re-scoring checkpoints
├── split_fingerprints.json  # sha256 of derived splits + retrieval caches
├── run_baselines.sh       # same-budget baseline matrix (resumable)
├── run_smoke.sh           # ~5 min end-to-end validation on synthetic data
├── run_real.sh            # initial real-data suite (historical 8-epoch protocol)
├── run_fullbudget.py      # full-epoch protocol driver
├── run_fullbudget_seeds.py  # multi-seed extension (in-flight cells)
├── run_gpu.py, run_all_gpu.py, run_gpu_yoochoose.py  # GPU fleet runners
├── make_gpu_package.py    # self-contained GPU upload package
├── fill_tables.py         # results JSON -> LaTeX \pmark replacement
├── local_patch.sh         # fifth seed + sensitivity sweeps (CPU, bash 3.2 safe)
├── make_release_v2.py     # rebuilds the submission artifact (full + lite)
└── README.md, NOTES.md, GPU_RUNBOOK.md
```

## Quick start

```bash
cd code
bash run_smoke.sh                 # ~5 min on CPU, validates everything
python tests/test_model.py        # unit tests only
```

## Running on the real benchmarks

Convert each public dataset (Diginetica, Gowalla, Yoochoose 1/64 and 1/4,
Nowplaying, RetailRocket) into `SessionData` pickles:

```python
from mplignn.data import SessionData, save_sessions
data = SessionData(train=[...], valid=[...], test=[...], n_items=...)
save_sessions(data, "data/diginetica.pkl")
```

then train and analyze:

```bash
python scripts/train.py --data data/diginetica.pkl --model mplignn \
    --epochs 50 --out results/diginetica_mplignn.json
for abl in no_residual no_global no_anchor softmax_attn no_denoise avg_intent; do
  python scripts/train.py --data data/diginetica.pkl --ablation $abl \
      --epochs 50 --out results/diginetica_$abl.json
done
python scripts/noise_injection.py --data data/diginetica.pkl \
    --ckpt results/diginetica_mplignn.pt --out results/diginetica_noise.json
python scripts/stratify_shift.py --data data/diginetica.pkl \
    --ckpt results/diginetica_mplignn.pt --out results/diginetica_stratify.json
python scripts/sensitivity.py --data data/diginetica.pkl \
    --out results/diginetica_sensitivity.json --epochs 50
python scripts/efficiency.py --data data/diginetica.pkl \
    --out results/diginetica_efficiency.json
```

`fill_tables.py` then populates the paper's placeholder cells:

```bash
python fill_tables.py --mapping results/mapping.json \
    --tex ../v1/sec_5_experiment.tex --in-place
```

## Model <-> paper correspondence

| Code | Paper |
|------|-------|
| `LocalEncoder` | Eq. (2)-(3), Section 4.2 |
| `GlobalEncoder` | Eq. (4)-(5), Section 4.3 (`vocab_blocks` = B) |
| `RelationEnhancement` | Eq. (6)-(10), Section 4.4 |
| `IntentHead` | Eq. (11)-(13), Section 4.5 |
| scoring + `F.cross_entropy` | Eq. (14)-(15), Section 4.6 |
| `entmax` | Peters et al. 2019, adaptive alpha in (1, 2] |

Ablation flags map one-to-one to the RQ2 variants:
`no_residual`, `no_global`, `no_anchor`, `softmax_attn`, `no_denoise`,
`avg_intent`.

## Implementation notes

1. **Global encoder scope.** The adaptive global encoder is applied to the
   unique items of each mini-batch rather than the full vocabulary; this is
   the practical realization of the block-wise approximation discussed in
   Section 4.3 and matches how global-graph baselines are implemented.
2. **EntMax gradients.** The bisection threshold is detached, so gradients
   flow through the piecewise-linear support (standard practice).
3. **Adaptive alpha.** `alpha = sigmoid(W t + b) + 1` clamped to (1.05, 2].
   Values near the lower bound fall back to softmax for stability.
4. **Relation inputs at evaluation.** The session-relation graph inputs
   `a_r`, `r0` are built identically at train and test time to avoid
   distribution shift.

## Integrity statement

`results/smoke_report.md` contains numbers produced **on synthetic data**
by `run_smoke.sh` on this machine. They validate the code and the
measurement protocols only. They are **not** benchmark results and must
never be copied into the manuscript tables. The manuscript's gray
placeholders must be filled exclusively from runs on the six public Web
benchmarks.

## GPU 全轮实验（2026-10 更新）

GPU 服务器一键流程，详见 `GPU_RUNBOOK.md`：

```bash
python3 make_gpu_package.py        # 打包 283MB 自包含上传包
# scp 上传 -> GPU 端: selftest -> run_gpu.py -> 回传 gpu_results.tar.gz
python3 scripts/ingest_gpu_results.py --tar gpu_results.tar.gz   # 回写+生成论文表格
```

关键文件：`run_gpu.py`（统一 GPU 运行器，yoo/full3 两种计划，断点续跑）、
`scripts/gpu_selftest.py`（7 项自检）、`scripts/ingest_gpu_results.py`
（校验+矩阵+LaTeX 表生成）。CUDA 修复见 `mplignn/data.py` 的
`batch_relation`（跨设备索引）与 `scripts/train.py` 的 Item-KNN 评估。
