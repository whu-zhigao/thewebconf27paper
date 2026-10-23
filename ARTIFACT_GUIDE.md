# Artifact Guide — *Not All Memory Gains Are Architectural: A Factorial Audit of Session-Based Recommendation*

Anonymized artifact for double-blind review (The Web Conference 2027).
The full variant ships the library code, every experiment driver, all 347
run records (JSON), the re-scoring checkpoints, split fingerprints, and
paper sources; the lite variant omits checkpoints and the compiled PDF so
it can accompany the OpenReview submission directly. **No number in the
paper is hand-typed**: every table cell traces to a released JSON record,
and one command re-checks 222 of them. During review, attach this archive
as supplementary material; no external link is required.

---

## 0. Contents at a glance

```
release_v2/
├── ARTIFACT_GUIDE.md            this file
├── MANIFEST.json                every file -> sha256 + size
├── SHA256SUMS                   shasum-compatible checksum list
├── attribution_check.yml        one-claim manifest for Appendix I
├── run_attribution_check.py     manifest-aware record/probe/effect audit
├── audit_report.sh              number audit + attribution audit
├── code/
│   ├── mplignn/                 model library (model, data, entmax, metrics, baselines)
│   ├── scripts/                 20 experiment drivers (train, pilot, audits, ...)
│   ├── analysis/                table generation + paper-number verification
│   │   ├── verify_paper_numbers.py     <-- one-command validation (222 checks)
│   │   ├── reidentify_checkpoint.py    <-- Appendix B as an executable
│   │   ├── make_tables.py              regenerates the factorial/stat tables
│   │   ├── run_cells.py                enumerates + reruns factorial cells
│   │   └── analysis_seeds.py           seed-level statistics
│   ├── tests/test_model.py      unit tests of the module properties claimed in §4
│   ├── results/real/            347 JSON records + 212 checkpoints
│   ├── split_fingerprints.json  sha256 of derived splits + retrieval caches
│   ├── run_fullbudget.py        full-epoch protocol driver (Table 5)
│   ├── run_fullbudget_seeds.py  optional future multi-seed extension
│   ├── run_baselines.sh         same-budget baseline matrix
│   └── README.md, NOTES.md, GPU_RUNBOOK.md, ...
└── paper/                       LaTeX sources + compiled PDF (revision 2)
```

**Two variants of this archive exist.** The *full* variant (~2.1 GB
unpacked) additionally contains the 212 re-scoring checkpoints
(`results/real/*.pt`) and the compiled paper PDF; the *lite* variant
(a few MB) omits both but keeps the complete LaTeX sources (rebuild the
PDF with §8 in about a minute). The ten-minute verification path below
runs unchanged in both variants **except step 3** (checkpoint
re-identification), which requires the full variant. Checkpoint-dependent
re-scoring paths (noise injection, read-out diagnostics) likewise need
the full variant plus rebuilt data (§5).

## 1. Requirements

- Python ≥ 3.9, PyTorch ≥ 2.0 (CPU is sufficient for all verification
  paths; a CUDA or MPS device only accelerates re-training)
- `pip install torch numpy pyyaml`
- ~20 min for the full verification path; disk as above
- The raw public datasets are **not** redistributed (their licenses);
  §5 explains the one-command rebuild and fingerprint check

## 2. The ten-minute verification path (no training)

```bash
# 0) integrity
shasum -a 256 -c SHA256SUMS          # expect: all OK

# 1) module properties claimed in the methodology section
cd code && python3 tests/test_model.py

# 2) 222 headline numbers plus the manifest-aware attribution audit
../audit_report.sh ../attribution_check.yml
#    -> "222/222 checks passed / ALL PAPER NUMBERS VERIFIED"
#    It validates record identity/configuration flags, recomputes the
#    paired HR@20 contrast, and runs both checkpoint probes.
#    The number audit covers: five-seed table (48 stats), factorial +
#    interaction (45,
#    incl. per-seed ix ranges and the MRR sign check), full-epoch table
#    (36), theory (24), audit (18), sensitivity (16), pilot (15),
#    noise (3), dead-module probe (1)

# 3) Appendix B (checkpoint re-identification) is also directly runnable
#    [full variant only -- lite omits checkpoints by design]
python3 analysis/reidentify_checkpoint.py
#    -> only the L=2 no-residual instantiation reproduces 30.09/21.35

# 4) regenerate the empirical LaTeX tables from the records
python3 analysis/make_tables.py      # writes analysis/tables_out/*.tex
```

Steps 2 and 4 need only the JSON records. Step 3 additionally needs
`code/data/` (§5): the split pickles and retrieval caches are required
at inference time; build them first (about 10–20 minutes for all five
datasets, Yoochoose the longest).

## 3. Claim → artifact map

| Paper item | Records (`code/results/real/`) | Regenerate with | Runtime |
|---|---|---|---|
| Pilot table (P1–P3) | `pilot_*.json` (5) | `scripts/pilot_study.py` | ~2 min/dataset |
| Factorial + paired tests (Tables 2–3) | `seed_mplignn*.json` (seeds 1–4; the L=2 residual cells) + `D*_{diginetica,gowalla}_s*.json` (six further cells per dataset, up to 10 seeds) | `analysis/run_cells.py` then `analysis/make_tables.py` | ~6 min/cell on CPU (12-epoch budget: NARM ≈ 2 min, MPLI-GNN ≈ 14 min) |
| Learning-rate sensitivity (App.) | `LRlow*.json` | `analysis/run_cells.py --lr-low` | as above |
| Theory table (λ, margin, τ, raw L*) | `theory_validation.json` | `scripts/theory_validation.py` (+ `calibration_transfer.py`) | ~5 min |
| Discriminating corpus | `dis_chain_*.json` | `scripts/make_synthetic.py` + `scripts/discriminating_test.py` | ~1 h (120 epochs) |
| Cross-architecture audit | `audit_srgnn_*.json`, `audit_gcegnn_*_{glob,noglob}.json` | `scripts/depth_gate_audit.py` | ~3 h (27 runs × ~7 min) |
| Same-budget matrix (reference only) | `<model>_<ds>.json` + `log_*.txt` | `run_baselines.sh` (resumable) | ~1–2 h total on CPU |
| Five-seed stability | `seed_{mplignn,mplignn_L1,srgnn,narm}_*_s[1-5].json` (60 runs) | inline loop in `GPU_RUNBOOK.md §seeds` | ~14 min/MPLI cell |
| Sensitivity sweep (App.) | `sensitivity_{diginetica,nowplaying}.json` | `scripts/sensitivity.py` | 34 runs ≈ 4 h |
| **Full-epoch table** | `full_*.json` (18 cells, seed 42) | `run_fullbudget.py` (+ `GPU_FULL=1` for Yoochoose-1/4) | Digi/Gowa ≈ 5.6–11.8 h/run, Yoo-1/4 ≈ 20–35 h/run on CPU; ~1/20 of that on GPU |
| Main-text Figures 2–3 and diagnostic plots | same records as the factorial / LR-sweep / full-epoch rows above | `paper/figures_editable/fig{2,3,4,5,6,7}_*.py` (matplotlib; recompute every plotted value from the JSONs and assert the paper's anchor numbers) | seconds each |
| Noise injection | `noise_*.json` | `scripts/noise_injection.py` | ~1 min/model |
| Shift strata | `stratify_diginetica.json` | `scripts/stratify_shift.py` | ~2 min |
| Read-out diagnostics | `readout_diginetica_L1res.json` | `scripts/diagnostics_readout.py` | ~2 min |
| Efficiency | per-run `sec_per_epoch`, `peak_rss_mb`, `params` fields | `scripts/efficiency.py` | ~30 min |
| Appendix B | `abl_no_residual_diginetica.pt` (full variant) | `analysis/reidentify_checkpoint.py` | ~1 min |

Every JSON records its configuration (model, flags, seed, lr, epochs),
the full validation/test metric block, per-epoch history, and wall-clock,
so any cell can be audited without re-running it.

## 4. Re-running an experiment family (examples)

```bash
cd code
# one budget-protocol cell (the paper's default configuration)
python3 scripts/train.py --data data/gowalla.pkl --model mplignn \
    --ablation no_residual --seed 7 --out results/real/D0R0_gowalla_s7.json

# the full-epoch protocol, one cell (Table 5)
python3 scripts/train.py --data data/diginetica.pkl --model mplignn \
    --ablation no_residual --lr 1e-3 --epochs 30 --patience 5 \
    --max-batches 0 --device cuda --out results/real/full_mplignn_L1_diginetica.json

# the multi-seed extension currently in flight (§7)
python3 run_fullbudget_seeds.py            # seeds 1,2 on Digi+Gowa
GPU_FULL=1 python3 run_fullbudget_seeds.py # adds Yoochoose-1/4
```

All drivers are resumable (existing `--out` files are skipped) and
device-adaptive (`cuda -> mps -> cpu`).

## 5. Datasets and fingerprints

Raw sources (public licenses, not redistributed): CIKM Cup 2016
(Diginetica), RecSys Challenge 2015 (Yoochoose), SNAP (Gowalla),
Nowplaying, Kaggle RetailRocket — each converter names its upstream
source in the header of `scripts/prepare_datasets.py` (and
`scripts/prepare_yoochoose.py`); fetch them from the official
challenge/repository pages.

```bash
python3 scripts/prepare_datasets.py        # diginetica, gowalla, nowplaying, retailrocket
python3 scripts/prepare_yoochoose.py       # builds the 1/64 and rebuilt 1/4 splits
python3 - <<'EOF'                          # compare against the shipped fingerprints
import hashlib, json
fp = json.load(open("split_fingerprints.json"))
for name, want in fp["splits"].items():
    got = hashlib.sha256(open(f"data/{name}","rb").read()).hexdigest()
    print(f"{name:28s}", "MATCH" if got == want else "DIFFERS")
EOF
```

`split_fingerprints.json` also pins the 18 retrieval-cache files
(`data/relcache/*.npz`), which the relation index rebuilds deterministically
from the splits. A mismatch anywhere means the rebuild diverged from the
submitted snapshot and should be reported before any other check.

## 6. Determinism expectations

Repeated identical runs agree to ±0.03 HR@20 points (floating-point
reduction order under CPU thread counts); **seed-level** spread is a
different, much larger quantity and is reported as mean±std in the paper
(see the bimodal Gowalla cell: two of five seeds train, three stop at
epoch one — that spread is a finding, not a bug). When re-running:
expect exact agreement for Item-KNN (no training), ±0.03 for re-scoring
saved checkpoints, and seed-spread-sized differences for re-training.

## 7. Future-work cells (outside the submitted claims)

The submitted paper claims no Yoochoose memory contrast, no second
architecture interaction, and no multi-seed full-training confirmation.
Those cells are future work: random/shuffled/frozen retrieval controls,
a complete GCE-GNN factorial, per-cell learning-rate grids beyond the
depth sensitivity reported here, and the multi-seed extension driven by
`run_fullbudget_seeds.py`. No table cell in the submitted PDF is a
placeholder, and the JSON records present are exactly those backing
printed numbers.

## 8. Rebuilding the paper PDF

The compiled PDF ships with the full variant; the lite variant carries
the same sources and rebuilds it with:

```bash
cd paper
pdflatex main_r2.tex && bibtex main_r2 && pdflatex main_r2.tex && pdflatex main_r2.tex
```

## 9. FAQ

- **Why do the HR@20 numbers not match published leaderboards?** The
  paper's comparison-context appendix answers this in detail: the standard
  SR-GNN-line splits contain ~4× more test sessions than ours, and
  Item-KNN — which trains nothing — already moves by several points
  between splits. All conclusions in the paper are protocol-internal.
- **Why does NARM's row carry a caveat?** The re-implementation diverges
  after its best epoch (per-batch loss > 10^4 on all six dataset files);
  the row reports the pre-divergence checkpoint selected by early
  stopping, as stated in the paper.
- **Why is Coverage not reported?** The released implementation computes
  it per batch, making the statistic batch-local; the paper drops it and
  explains why (appendix).
- **Checkpoint provenance.** Every `.pt` in `results/real/` pairs with a
  same-named `.json` recording its configuration and metrics; the
  factorial-cell checkpoints are omitted for size (their JSON records
  are complete), and all re-scoring checkpoints are included in the
  full variant.
