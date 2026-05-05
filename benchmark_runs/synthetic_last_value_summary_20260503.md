# Synthetic Last-Value Tracking Summary

Date: 2026-05-03

Task: train at context 128, then evaluate last written value at longer contexts.
All main runs use 2 layers, d_model 128, vocab 128, 16 values, 1-4 writes, batch 64, 2000 steps, BF16 on one NVIDIA L4.

## Main 3-Seed Results

| Context | Gated MOGT acc | Transformer RoPE acc | Gap |
|---:|---:|---:|---:|
| 128 | 100.00% +/- 0.00% | 99.48% +/- 0.25% | 0.52 pp |
| 256 | 98.26% +/- 3.02% | 70.25% +/- 13.68% | 28.01 pp |
| 512 | 89.49% +/- 15.11% | 22.88% +/- 8.35% | 66.60 pp |
| 1024 | 69.71% +/- 18.69% | 10.51% +/- 1.24% | 59.20 pp |

## NoPE Transformer Matched-Eval 3-Seed Baseline

These runs remove RoPE and use the same 2048 eval examples per context as the main table.

| Context | Transformer NoPE acc | Eval examples / seed |
|---:|---:|---:|
| 128 | 99.07% +/- 0.32% | 2048/2048/2048 |
| 256 | 98.10% +/- 0.85% | 2048/2048/2048 |
| 512 | 91.47% +/- 5.80% | 2048/2048/2048 |
| 1024 | 68.64% +/- 15.51% | 2048/2048/2048 |

## NoPE Transformer Earlier Light-Eval Baseline

These runs remove RoPE and use 128 eval examples per context. They are not directly mixed into the main table because the eval protocol is lighter, but they answer whether the RoPE baseline was artificially weak.

| Context | Transformer NoPE acc |
|---:|---:|
| 128 | 97.66% +/- 0.00% |
| 256 | 98.18% +/- 1.19% |
| 512 | 92.71% +/- 7.98% |
| 1024 | 65.36% +/- 12.41% |
| 2048 | 37.24% +/- 14.43% |
| 4096 | 12.24% +/- 11.11% |

## Seed-42 Long Extrapolation

These runs train at context 128 and evaluate farther out with 64-128 examples per context.

| Model | 128 | 1024 | 2048 | 4096 | 8192 | Artifact |
|---|---:|---:|---:|---:|---:|---|
| Gated MOGT | 100.00% | 82.81% | 67.19% | 34.38% | 21.88% | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps2000_eval8192.json` |
| Transformer RoPE | 100.00% | 3.12% | 9.38% | 4.69% | 4.69% | `benchmark_runs/synthetic_last_value_transformer_ctx128_seed42_steps2000_eval8192.json` |
| Transformer RoPE theta=1e6 | 99.22% | 8.59% | 8.59% | 3.91% | - | `benchmark_runs/synthetic_last_value_transformer_rope1e6_ctx128_seed42_steps2000.json` |
| Transformer NoPE | 97.66% | 57.81% | 28.91% | 7.03% | - | `benchmark_runs/synthetic_last_value_transformer_nope_ctx128_seed42_steps2000.json` |

## 1024-Context Individual Seeds

| Model | Seed | Accuracy | Loss | Artifact |
|---|---:|---:|---:|---|
| Gated MOGT | 7 | 49.12% | 3.7386 | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed7_steps2000.json` |
| Gated MOGT | 42 | 85.60% | 1.3569 | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps2000.json` |
| Gated MOGT | 123 | 74.41% | 3.0351 | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed123_steps2000.json` |
| Transformer RoPE | 7 | 11.52% | 7.7955 | `benchmark_runs/synthetic_last_value_transformer_ctx128_seed7_steps2000.json` |
| Transformer RoPE | 42 | 10.89% | 4.1543 | `benchmark_runs/synthetic_last_value_transformer_ctx128_seed42_steps2000.json` |
| Transformer RoPE | 123 | 9.13% | 5.3461 | `benchmark_runs/synthetic_last_value_transformer_ctx128_seed123_steps2000.json` |

## Train-512 Dense 3-Seed Results

Both models use dense state supervision, train context 512, 2000 steps, batch 16, and 64 eval examples per context.

| Context | MOGT coupled write-forget dense | MOGT identity dual-gate dense | Transformer NoPE dense | Coupled gap vs NoPE |
|---:|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 97.40% +/- 0.90% | 2.60 pp |
| 1024 | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 98.44% +/- 1.56% | 1.56 pp |
| 2048 | 100.00% +/- 0.00% | 94.27% +/- 8.61% | 95.31% +/- 2.71% | 4.69 pp |
| 4096 | 100.00% +/- 0.00% | 70.83% +/- 7.05% | 90.10% +/- 5.49% | 9.90 pp |
| 8192 | 100.00% +/- 0.00% | 47.40% +/- 8.88% | 78.65% +/- 12.63% | 21.35 pp |

## Coupled Write-Forget Long Probe

Seed 42 only, trained at context 512. This probe uses fewer eval examples than the main table and is a stress test rather than a final multi-seed result.

| Context | Accuracy | Eval examples |
|---:|---:|---:|
| 512 | 100.00% | 16 |
| 8192 | 100.00% | 16 |
| 16384 | 100.00% | 16 |
| 32768 | 100.00% | 16 |
| 65536 | 100.00% | 16 |

Artifact: `benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000_eval65536.json`. Peak memory 1234.4 MB; train elapsed 60.4s.

## Coupled Gate Diagnostics

Seed 42, train context 512. Gate values are means on a fresh training-context batch after training.

| Block | SET | value token | filler | QUERY |
|---:|---:|---:|---:|---:|
| 0 | 0.0001% | 88.83% | 0.0001% | 0.0001% |
| 1 | 42.15% | 26.97% | 39.26% | 37.56% |

Artifact: `benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000.json`.

## Train-512 Scaling Probe

Seed 42 only. This is a scale stress test, not a final result.

| Variant | 512 | 1024 | 2048 | 4096 | 8192 | Peak MB | Train elapsed | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gated MOGT direct train512 | 12.50% | 7.81% | 3.12% | 10.94% | 6.25% | 522.2 | 61.5s | `benchmark_runs/synthetic_last_value_mogt_gated_ctx512_seed42_steps2000.json` |
| Gated MOGT curriculum to 512 | 39.06% | 34.38% | 6.25% | 17.19% | 6.25% | 522.2 | 81.4s | `benchmark_runs/synthetic_last_value_mogt_gated_curriculum_ctx512_seed42_steps3000.json` |
| MOGT Cayley dual-gate dense train512 | 100.00% | 100.00% | 79.69% | 60.94% | 40.62% | 762.8 | 69.5s | `benchmark_runs/synthetic_last_value_mogt_dualgate_dense_ctx512_seed42_steps2000.json` |
| MOGT identity dual-gate dense train512, 400 steps | 100.00% | 82.81% | 56.25% | 31.25% | 21.88% | 604.7 | 15.1s | `benchmark_runs/synthetic_last_value_mogt_identity_dualgate_dense_ctx512_seed42_steps400.json` |
| MOGT identity dual-gate dense train512, 2000 steps | 100.00% | 100.00% | 100.00% | 78.12% | 54.69% | 605.9 | 62.4s | `benchmark_runs/synthetic_last_value_mogt_identity_dualgate_dense_ctx512_seed42_steps2000.json` |
| MOGT identity dual-gate keep12 dense train512 | 100.00% | 96.88% | 87.50% | 46.88% | 37.50% | 605.9 | 32.2s | `benchmark_runs/synthetic_last_value_mogt_identity_dualgate_keep12_dense_ctx512_seed42_steps1000.json` |
| MOGT identity dual-gate damp1.0002 dense train512 | 100.00% | 89.06% | 70.31% | 35.94% | 31.25% | 605.9 | 32.3s | `benchmark_runs/synthetic_last_value_mogt_identity_dualgate_damp10002_dense_ctx512_seed42_steps1000.json` |
| MOGT identity value-gate-only dense train512 | 87.50% | 92.19% | 64.06% | 48.44% | 25.00% | 545.4 | 55.8s | `benchmark_runs/synthetic_last_value_mogt_identity_valuegate_only_dense_ctx512_seed42_steps2000.json` |
| MOGT identity coupled write-forget dense train512 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 593.5 | 60.5s | `benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000.json` |
| MOGT identity forget-ReLU dual-gate dense train512 | 87.50% | 85.94% | 68.75% | 34.38% | 4.69% | 606.2 | 32.8s | `benchmark_runs/synthetic_last_value_mogt_identity_forgetrelu_dualgate_dense_ctx512_seed42_steps1000.json` |
| MOGT identity residual transport-gate dense train512 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 606.0 | 32.4s | `benchmark_runs/synthetic_last_value_mogt_identity_resgate_dualgate_dense_ctx512_seed42_steps1000.json` |
| Transformer NoPE dense train512, 400 steps | 89.06% | 76.56% | 76.56% | 76.56% | 48.44% | 438.6 | 8.2s | `benchmark_runs/synthetic_last_value_transformer_nope_dense_ctx512_seed42_steps400.json` |
| Transformer NoPE dense train512, 2000 steps | 98.44% | 100.00% | 96.88% | 89.06% | 82.81% | 438.7 | 29.5s | `benchmark_runs/synthetic_last_value_transformer_nope_dense_ctx512_seed42_steps2000.json` |
| Transformer NoPE direct train512 | 100.00% | 96.88% | 90.62% | 59.38% | 32.81% | 328.5 | 27.5s | `benchmark_runs/synthetic_last_value_transformer_nope_ctx512_seed42_steps2000.json` |

## Seed-42 Ablation

| Variant | 128 | 256 | 512 | 1024 | Peak MB | Train elapsed | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| MOGT, hybrid, 500 steps | 6.54% | 5.57% | 6.88% | 6.64% | 542.9 | 16.8s | `benchmark_runs/synthetic_last_value_mogt_ctx128_seed42_steps500.json` |
| MOGT, sequential, 500 steps | 5.86% | 5.66% | - | - | 250.7 | 51.5s | `benchmark_runs/synthetic_last_value_mogt_sequential_ctx128_seed42_steps500.json` |
| Gated MOGT, 500 steps | 13.18% | 10.79% | 9.03% | 8.74% | 542.9 | 17.6s | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps500.json` |
| Gated MOGT, 2000 steps | 100.00% | 100.00% | 99.66% | 85.60% | 542.9 | 63.9s | `benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps2000.json` |
| Transformer RoPE, 2000 steps | 99.76% | 79.05% | 32.47% | 10.89% | 314.0 | 42.9s | `benchmark_runs/synthetic_last_value_transformer_ctx128_seed42_steps2000.json` |
| Transformer NoPE, 2000 steps | 97.66% | 98.44% | 96.09% | 57.81% | 176.0 | 39.0s | `benchmark_runs/synthetic_last_value_transformer_nope_ctx128_seed42_steps2000.json` |

## Interpretation

- Original MOGT stays near chance on this overwrite-style memory task.
- Adding a token-dependent transport gate makes the recurrence learnable and gives much stronger length extrapolation than the matched Transformer in this small setting.
- Removing RoPE makes Transformer a much stronger extrapolation baseline. Under the matched 2048-example eval protocol, NoPE Transformer reaches 68.64% +/- 15.51% at 1024, close to the gated MOGT main-table mean of 69.71% +/- 18.69%. Future claims must treat NoPE/position-robust attention as a first-class baseline rather than a caveat.
- Train512 exposed two separate issues: Cayley/Magnus transport makes overwrite memory hard to optimize, while identity transport plus dual gates and dense state supervision learns quickly.
- Coupling the value/write gate to transport forgetting is the strongest current mechanism: across seeds 7/42/123 it reaches 100.00% +/- 0.00% from 512 through 8192 after training only at context 512.
- The value-gate-only ablation learns in-distribution but drops to 48.44% at 4096 and 25.00% at 8192, so explicit transport/forget gating is still needed for state preservation.
- The residual transport-gate mode was numerically unstable in the seed-42 probe, while forget-ReLU was stable but much worse at far extrapolation.
- Under dense train512 supervision, independently gated identity MOGT learns faster early but trails NoPE at far context; coupled write-forget MOGT fixes the single-slot far-extrapolation gap while still using more memory and time than the small NoPE Transformer implementation.
- This is not yet a language-modeling win. It is a scoped synthetic result that identifies a plausible paper direction: gated affine transport for recurrent state tracking and length extrapolation.
- Next evidence needed: larger contexts, more tasks, training-token matched curves, throughput/memory tradeoffs, and LM hybrid experiments.
