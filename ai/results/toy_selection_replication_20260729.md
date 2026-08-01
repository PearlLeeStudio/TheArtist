# Toy selection-replication run record (2026-07-29)

Synthetic second-domain replication of the checkpoint-selection failure, for the ICASSP 2027
paper's hardening pass (bar-review lever: converts "incident at one lab" into "reproducible
phenomenon"). Script: `ai/analysis/toy_selection_replication.py`. Data:
`toy_selection_replication_20260729.csv` (per seed × epoch: src/tgt val losses, lr, hash16,
is_init, analytic mixed-val at source shares 0.50/0.87/0.98).

## Setup

- TinyLM: 2-layer causal transformer encoder, d=64, 4 heads, 32-token vocab, seq len 64.
- Source/target: two fixed order-1 Markov transition matrices (generator seed 1000, drawn
  sequentially; genuinely different chains over the same vocabulary).
- Pretrain: 30 epochs Adam 1e-3 on 8,000 source sequences (to source convergence — the
  incident's precondition; the v1 failure requires the init to sit at the source optimum).
- Fine-tune: resume at epoch 3, epochs 3–9, on 8,000 source + 1,200 target (87/13 rehearsal
  mix, mirroring F1). LR schedule: 2-epoch warmup from zero, stepped per epoch → epoch 3 runs
  entirely at lr=0 (Adam; zero parameter update, moments may accumulate).
- Seeds: 42, 123, 7. All CPU (WSL), minutes-scale.
- Mixed validation at source share s is computed analytically as s·L_src + (1−s)·L_tgt from
  the per-epoch src/tgt val losses (exact for equal-length sequences), so selection can be
  evaluated at ANY composition post hoc without retraining.

## Results

1. **Zero-update epoch reproduces 3/3**: epoch-3 checkpoint hash == init hash in every seed
   (bit-identical initialization enters the candidate pool).
2. **Selection is a function of validation composition** (analytic sweep over s):

   | seed | s≤0.95 | s=0.98–0.998 | s=0.999 | s=1.0 |
   |---|---|---|---|---|
   | 42 | ep9 | ep8 | **ep3 = INIT** | **ep3 = INIT** |
   | 123 | ep9 | ep9 | ep9 | **ep3 = INIT** |
   | 7 | ep9 | ep9 | **ep3 = INIT** | **ep3 = INIT** |

   The pick marches backward as the source share grows and reaches the bit-identical
   initialization at s ≥ 0.999 in 2/3 seeds and at s = 1.0 in 3/3.
3. **The flip threshold follows the convex-combination decomposition** (threshold where
   s·Δ_src-degradation overtakes (1−s)·Δ_tgt-gain). Toy endpoint deltas: Δsrc ≈ +0.005–0.010
   nats vs Δtgt ≈ −5.3 to −5.5 nats (target near-unlearnable-before → strongly learned), so
   the toy threshold sits near s ≈ 0.999. Back-computed from the F1v2 6-src run's endpoint
   losses (mixed val 0.5578→0.5929, jazz val 1.4213→0.9482, jazz val share 179/10,179 =
   1.76%): implied ΔL_src ≈ +0.044, threshold s* ≈ 0.91 — well below that run's actual 0.982
   source share, which is why the incident's rule landed on the init.
4. **Honest difference from the incident**: the toy's source loss is NOT monotone (worst at
   epoch 4, partially re-recovered by rehearsal: e.g. seed 42: 1.4316 → 1.4893 → 1.4420), so
   the incident's clean monotone mixed-val rise is a property of its regime, not a universal
   signature. What replicates unconditionally is the load-bearing pair: the init enters the
   pool via the zero-update epoch, and the validation composition decides whether it gets
   selected.

## History

- v1 run (same date, superseded): SGD lr 3e-4, 8 pretrain epochs (not converged) → mixed val
  kept falling at all tested mixtures; both criteria picked ep9. Confirmed the convergence
  precondition and motivated the v2 design (Adam, 30-epoch pretrain, analytic s-sweep).
