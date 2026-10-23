# Baseline Re-implementation Notes (NOTES.md)

All twelve neural baselines in Table 4 of the paper are re-implemented in
`mplignn/baselines.py` and trained by the identical driver
(`scripts/train.py`) with the identical data loader, splits, full-sort
evaluation, loss (softmax cross-entropy over the full vocabulary),
optimizer, and CPU budget as MPLI-GNN: d=64, lr 5e-4, 12 epochs, at most
300 batches/epoch, patience 4.  Table cells therefore measure
**our shared-budget re-implementations**, not the published numbers.

Per-model notes (architecture kept where stated; simplifications listed):

| Model | Kept from the original | Simplification |
|---|---|---|
| FPMC | factorized transition + user bias | anonymous session -> mean item embedding as user factor; CE instead of BPR |
| GRU4Rec | GRU encoder, last-state read-out | packed sequences instead of session-parallel buckets; CE instead of sampled ranking loss |
| NARM | GRU + published attention read-out | single GRU layer |
| STAMP | MLP on [last item; session mean] + residual | — |
| LESSR | graph convolution over ordered session adjacency | gated GGNN + positional embeddings approximate the edge-order graph |
| DHCN | session-item hypergraph propagation | one round; auxiliary self-supervision loss dropped |
| GCE-GNN | local GGNN + tanh global attention over level-transformed embeddings | position encoding simplified to identity; notably weak under this budget (5.05/2.99 on Diginetica) — flagged in the paper |
| MTD | hierarchical GGNN | degree-normalized adjacency; mean+last gated read-out |
| DSAN | alpha-EntMax sparse self-attention over the prefix | one layer; dual-stream realized via the shared vocabulary embedding |
| MSGIFSR | intent units at granularities {1,2,4} | contiguous-block means fused by a small GRU |
| MSGAT | bi-channel entmax attention (local + global) | global channel over batch-unique items |

Reproduce the full matrix:

```bash
cd code && ./run_baselines.sh     # resumable; skips existing JSONs
```

Noise robustness (Table 7):

```bash
for m in srgnn gcegnn dsan msgat; do
  python scripts/noise_injection.py --data data/diginetica.pkl \
    --ckpt results/real/${m}_diginetica.pt --model $m \
    --out results/real/noise_${m}_diginetica.json
done
```
