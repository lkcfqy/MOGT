# Synthetic Multi-Slot Tracking Summary

Date: 2026-05-03

Task: the sequence starts with a tracked slot id, then contains writes to
multiple slots. The model must return the last value written to the tracked
slot. This is a harder selective state-routing task than single-slot
last-value tracking.

## Slot-Addressed 2-Slot Dense 3-Seed Result

Both models use train context 512, 2 slots, 16 values, 2-6 writes, dense state
supervision, batch 16, 3000 steps, and 64 eval examples per context. The
slot-addressed MOGT gate uses `current_prev_prefix`: current token, previous
token, and the tracked slot token at prefix position 1.

| Context | Slot-addressed MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 45.31% +/- 4.69% | 54.69 pp |
| 1024 | 100.00% +/- 0.00% | 41.15% +/- 3.93% | 58.85 pp |
| 2048 | 98.44% +/- 1.56% | 34.90% +/- 3.25% | 63.54 pp |
| 4096 | 86.46% +/- 3.25% | 44.79% +/- 9.42% | 41.67 pp |

## Slot-Addressed 4-Slot Dense 3-Seed Result

Both models use train context 512, 4 slots, 16 values, 4-12 writes, dense state
supervision, batch 16, 3000 steps, and 64 eval examples per context.

| Context | Slot-addressed MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 23.96% +/- 3.61% | 76.04 pp |
| 1024 | 100.00% +/- 0.00% | 25.00% +/- 4.13% | 75.00 pp |
| 2048 | 100.00% +/- 0.00% | 21.35% +/- 4.77% | 78.65 pp |
| 4096 | 98.96% +/- 0.90% | 24.48% +/- 7.05% | 74.48 pp |

## Slot-Addressed 2-Slot Final-Only 3-Seed Result

Both models use train context 512, 2 slots, 16 values, 2-6 writes, final query
supervision only, batch 16, 3000 steps, and 64 eval examples per context.

| Context | Slot-addressed MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 42.19% +/- 2.71% | 57.81 pp |
| 1024 | 100.00% +/- 0.00% | 40.62% +/- 7.16% | 59.38 pp |
| 2048 | 100.00% +/- 0.00% | 47.92% +/- 3.93% | 52.08 pp |
| 4096 | 96.35% +/- 3.25% | 42.71% +/- 21.67% | 53.65 pp |

## Slot-Addressed 4-Slot Final-Only Slot-Curriculum 3-Seed Result

Both models use train context 512, 4 slots, 16 values, 4-12 writes, final query
supervision only, batch 16, 3000 steps, and 64 eval examples per context. The
training generator starts with 2 active slots and increases to all 4 slots over
the first 1500 steps.

| Context | Slot-addressed MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 31.25% +/- 7.16% | 68.75 pp |
| 1024 | 100.00% +/- 0.00% | 29.69% +/- 3.12% | 70.31 pp |
| 2048 | 99.48% +/- 0.90% | 22.40% +/- 7.05% | 77.08 pp |
| 4096 | 94.27% +/- 2.39% | 18.23% +/- 11.73% | 76.04 pp |

## Slot-Addressed 6-Slot Final-Only Slot-Curriculum 3-Seed Result

Both models use train context 512, 6 slots, 16 values, 6-18 writes, final query
supervision only, batch 16, 4000 steps, and 64 eval examples per context. The
generator starts with 2 active slots and increases to all 6 slots over the
first 2000 steps.

| Context | Slot-addressed MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 20.83% +/- 1.80% | 79.17 pp |
| 1024 | 100.00% +/- 0.00% | 22.92% +/- 5.02% | 77.08 pp |
| 2048 | 98.96% +/- 0.90% | 21.35% +/- 5.49% | 77.60 pp |
| 4096 | 96.88% +/- 2.71% | 21.88% +/- 8.27% | 75.00 pp |

## Slot-Addressed 8-Slot Final-Only Slot-Curriculum 3-Seed Result

Both models use train context 512, 8 slots, 16 values, 8-24 writes, final query
supervision only, batch 16, 5000 steps, and 64 eval examples per context. The
generator starts with 2 active slots and increases to all 8 slots over the
first 2500 steps.

| Context | Slot-addressed MOGT | Transformer NoPE | HF Mamba d128 | HF Mamba d192 |
|---:|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 17.19% +/- 2.71% | 18.23% +/- 3.61% | 17.71% +/- 8.02% |
| 1024 | 100.00% +/- 0.00% | 16.15% +/- 6.51% | 13.02% +/- 1.80% | 13.54% +/- 6.51% |
| 2048 | 97.40% +/- 0.90% | 22.92% +/- 5.02% | 13.54% +/- 7.86% | 19.27% +/- 7.71% |
| 4096 | 85.42% +/- 11.93% | 10.94% +/- 2.71% | 18.23% +/- 1.80% | 16.15% +/- 6.31% |

The 8-slot result remains positive but shows real extrapolation variance:
individual MOGT seeds reach 98.44%, 75.00%, and 82.81% at 4096. This is a
useful scaling boundary for the paper rather than a solved-all-settings claim.
The HF-Mamba baselines are stronger than NoPE at 4096 in these small runs but
still far below MOGT: d128 reaches 18.23% +/- 1.80%, and parameter-matched d192
reaches 16.15% +/- 6.31%.

Curriculum ablation, seed 42 only: direct 6-slot final-only MOGT without active
slot curriculum reaches 23.44% at 4096, compared with 95.31% for the
curriculum run at the same seed. This indicates that the 6-slot result depends
on an optimization curriculum, not just the static architecture.

8-slot curriculum ablation, seed 42 only: direct 8-slot final-only MOGT without
active slot curriculum reaches 12.50% at 4096, compared with 98.44% for the
curriculum run at the same seed. The direct run's gate diagnostic is not
slot-selective: block 0 matched and unmatched value gates are both about 44.3%,
while SET/query/slot/filler are all about 49%.

## Scratch GRU 6-Slot Early Probe

This is an early-learning probe rather than a full baseline. A 2-layer
`d_model=128` GRU with tied embeddings, trained on the same 6-slot final-only
task for 500 steps with the same 2-to-6-slot curriculum schedule, reaches only
6.25% train accuracy at step 500 and 6.25% eval accuracy at 4096. The unfused
CUDA GRU path is slow in this environment: 500 steps took 228.1s. A full
recurrent-baseline comparison remains open and should use a fused/optimized
implementation.

Artifact:
`benchmark_runs/synthetic_multislot6_gru_slotcurr2000_finalonly_ctx512_seed42_steps500.json`.

## Weaker 4-Slot Supervision Probes

Seed 42 unless noted.

| Supervision | Model | 512 | 1024 | 2048 | 4096 | Artifact |
|---|---|---:|---:|---:|---:|---|
| Final-only | Slot-addressed MOGT | 56.25% | 45.31% | 56.25% | 53.12% | `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_finalonly_ctx512_seed42_steps3000.json` |
| Final-only | Transformer NoPE | 28.12% | 18.75% | 21.88% | 28.12% | `benchmark_runs/synthetic_multislot4_transformer_nope_finalonly_ctx512_seed42_steps3000.json` |
| Write-only, 3 seeds | Slot-addressed MOGT | 56.25% +/- 39.03% | 54.69% +/- 40.53% | 53.12% +/- 41.66% | 58.85% +/- 37.05% | `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_writeloss_ctx512_seed*_steps3000.json` |
| Dense 1500 then final-only | Slot-addressed MOGT | 100.00% | 100.00% | 100.00% | 96.88% | `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_dense1500_final_ctx512_seed42_steps3000.json` |
| Dense 1500 then final-only | Transformer NoPE | 21.88% | 20.31% | 20.31% | 26.56% | `benchmark_runs/synthetic_multislot4_transformer_nope_dense1500_final_ctx512_seed42_steps3000.json` |
| Slot curriculum final-only, 3 seeds | Slot-addressed MOGT | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 99.48% +/- 0.90% | 94.27% +/- 2.39% | `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_ctx512_seed*_steps3000.json` |
| Slot curriculum final-only, 3 seeds | Transformer NoPE | 31.25% +/- 7.16% | 29.69% +/- 3.12% | 22.40% +/- 7.05% | 18.23% +/- 11.73% | `benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_ctx512_seed*_steps3000.json` |

## Slot-Addressed Gate Diagnostic

Seed 42, 2-slot, 1500 steps. The diagnostic computes the mean rank-wise value
gate on a fresh training-context batch. Block 0 is selective: it opens much
more on value tokens whose preceding slot matches the tracked prefix slot.

| Block | matched value | unmatched value | SET | slot | filler | QUERY |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32.64% | 2.04% | 0.02% | 0.17% | 0.04% | 0.02% |
| 1 | 21.77% | 24.38% | 21.75% | 19.82% | 20.37% | 25.47% |

Artifact: `benchmark_runs/synthetic_multislot2_mogt_slotaddr_coupled_rank_value_forget_dense_ctx512_seed42_steps1500_diag.json`.

## Seed-42 2-Slot Dense Probe

Both models use train context 512, 2 slots, 16 values, 2-6 writes, dense state
supervision, batch 16, BF16 on one NVIDIA L4.

| Model | Steps | 512 | 1024 | 2048 | 4096 | Peak MB | Train elapsed | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MOGT identity dual-gate | 1500 | 45.31% | 56.25% | 53.12% | 48.44% | 604.2 | 71.7s | `benchmark_runs/synthetic_multislot2_mogt_identity_dualgate_dense_ctx512_seed42_steps1500.json` |
| Transformer NoPE | 1500 | 31.25% | 34.38% | 34.38% | 32.81% | 436.9 | 55.6s | `benchmark_runs/synthetic_multislot2_transformer_nope_dense_ctx512_seed42_steps1500.json` |
| MOGT identity dual-gate | 3000 | 54.69% | 50.00% | 48.44% | 32.81% | 604.3 | 142.1s | `benchmark_runs/synthetic_multislot2_mogt_identity_dualgate_dense_ctx512_seed42_steps3000.json` |
| MOGT identity rank-gate | 3000 | 53.12% | 53.12% | 50.00% | 60.94% | 613.4 | 98.9s | `benchmark_runs/synthetic_multislot2_mogt_identity_rankgate_dense_ctx512_seed42_steps3000.json` |
| MOGT coupled scalar write-forget | 3000 | 56.25% | 54.69% | 48.44% | 59.38% | 594.2 | 156.9s | `benchmark_runs/synthetic_multislot2_mogt_identity_coupled_scalar_value_forget_dense_ctx512_seed42_steps3000.json` |
| MOGT coupled rank write-forget | 3000 | 54.69% | 53.12% | 51.56% | 67.19% | 599.9 | 156.6s | `benchmark_runs/synthetic_multislot2_mogt_identity_coupled_rank_value_forget_dense_ctx512_seed42_steps3000.json` |
| MOGT slot-addressed coupled rank | 3000 | 100.00% | 100.00% | 96.88% | 89.06% | 635.6 | 99.6s | `benchmark_runs/synthetic_multislot2_mogt_slotaddr_coupled_rank_value_forget_dense_ctx512_seed42_steps3000.json` |
| Transformer NoPE | 3000 | 50.00% | 40.62% | 35.94% | 54.69% | 437.1 | 111.7s | `benchmark_runs/synthetic_multislot2_transformer_nope_dense_ctx512_seed42_steps3000.json` |

## Seed-42 4-Slot Dense Probe

Both models use train context 512, 4 slots, 16 values, 4-12 writes, dense state
supervision, batch 16, 1000 steps.

| Model | 512 | 1024 | 2048 | 4096 | Peak MB | Train elapsed | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| MOGT identity dual-gate | 21.88% | 37.50% | 34.38% | 23.44% | 605.8 | 66.0s | `benchmark_runs/synthetic_multislot_mogt_identity_dualgate_dense_ctx512_seed42_steps1000.json` |
| Transformer NoPE | 20.31% | 26.56% | 26.56% | 18.75% | 437.1 | 55.9s | `benchmark_runs/synthetic_multislot_transformer_nope_dense_ctx512_seed42_steps1000.json` |

## Interpretation

- Unconditioned multi-slot routing is not solved by the earlier MOGT variants.
- Adding prefix-conditioned slot-addressing to the coupled write-forget gate
  changes the result sharply: 2-slot tracking reaches 100/100/98.44/86.46
  across 512/1024/2048/4096 over three seeds.
- The 4-slot result is also strong across three seeds, reaching
  100/100/100/98.96 across 512/1024/2048/4096 while a matched NoPE Transformer
  stays near 21%-25%.
- This supports the next paper claim: coupled write-forget solves single-slot
  memory, and prefix-conditioned gate inputs make affine transport viable for
  tracked multi-slot state routing, first under dense supervision and then
  under final-query-only supervision with an active-slot curriculum.
- The 2-slot task no longer requires dense state supervision: final-query-only
  training reaches 96.35% +/- 3.25% at 4096 across three seeds.
- Direct 4-slot final-only training is weak, but a slot-count curriculum
  changes the final-only 4-slot result sharply: MOGT reaches
  94.27% +/- 2.39% at 4096 across three seeds, while the matched NoPE
  Transformer reaches 18.23% +/- 11.73%.
- The 6-slot final-only curriculum result is also strong across three seeds:
  MOGT reaches 96.88% +/- 2.71% at 4096, while the matched NoPE Transformer
  reaches 21.88% +/- 8.27%.
- The 8-slot final-only curriculum result is still positive but more variable:
  MOGT reaches 85.42% +/- 11.93% at 4096, while NoPE reaches
  10.94% +/- 2.71%, HF Mamba d128 reaches 18.23% +/- 1.80%, and
  parameter-matched HF Mamba d192 reaches 16.15% +/- 6.31%.
- Direct 6-slot final-only without curriculum fails in the seed-42 ablation
  (23.44% at 4096), showing that the curriculum is an important optimization
  condition.
- Direct 8-slot final-only without curriculum also fails in the seed-42
  ablation (12.50% at 4096), and its gate diagnostic does not separate matched
  from unmatched values.
- A scratch GRU early-learning probe also fails to learn within 500 steps, but
  it is not a complete recurrent baseline because the unfused implementation is
  much slower than the MOGT/Transformer runs.
- Remaining gaps: direct 4-slot/6-slot final-only without curriculum, stronger
  recurrent baselines, larger slot counts, and non-synthetic language-modeling
  or hybrid experiments.
