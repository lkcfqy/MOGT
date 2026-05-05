# Results Snapshot

Last updated: 2026-05-05

This file is generated from `mogt-experiment-v1` standard reports by
`summarize_paper_results.py`.

## Standard Evidence Status

- Standard report index: `benchmark_runs/standard_report_index.md`
- Last-value summary: `benchmark_runs/synthetic_last_value_summary_20260503.md`
- Multi-slot summary: `benchmark_runs/synthetic_multislot_standard_summary_20260504.md`
- Standard reports loaded into this snapshot: 118 ok reports.

## Main Table A: Single-Slot Overwrite State Tracking

| Context | Coupled MOGT | Transformer NoPE | Gap |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 97.40% +/- 0.90% | 2.60 pp |
| 1024 | 100.00% +/- 0.00% | 98.44% +/- 1.56% | 1.56 pp |
| 2048 | 100.00% +/- 0.00% | 95.31% +/- 2.71% | 4.69 pp |
| 4096 | 100.00% +/- 0.00% | 90.10% +/- 5.49% | 9.90 pp |
| 8192 | 100.00% +/- 0.00% | 78.65% +/- 12.63% | 21.35 pp |

## Main Table B: Tracked 4-Slot Final-Query Routing

| Context | Slot-addressed MOGT | Transformer NoPE | HF-Mamba d192 |
|---:|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 31.25% +/- 5.41% | 17.71% +/- 10.97% |
| 1024 | 100.00% +/- 0.00% | 29.69% +/- 5.63% | 20.83% +/- 9.92% |
| 2048 | 99.48% +/- 0.90% | 23.44% +/- 5.63% | 15.62% +/- 5.63% |
| 4096 | 94.27% +/- 2.39% | 21.35% +/- 6.31% | 19.79% +/- 10.97% |

## Ablation: Curriculum Is An Optimization Condition

| Context | MOGT direct | NoPE direct | MOGT curriculum |
|---:|---:|---:|---:|
| 512 | 44.27% +/- 14.52% | 26.56% +/- 2.71% | 100.00% +/- 0.00% |
| 1024 | 38.54% +/- 21.89% | 24.48% +/- 3.93% | 100.00% +/- 0.00% |
| 2048 | 41.67% +/- 22.61% | 21.35% +/- 3.25% | 99.48% +/- 0.90% |
| 4096 | 44.79% +/- 17.21% | 25.00% +/- 5.63% | 94.27% +/- 2.39% |

## Systems Snapshot: Core Operator Timing

Core timing only: affine scan excludes connection/value projection,
matrix exponential/Cayley construction, normalization, FFN, and LM head;
attention timing measures FlashAttention/SDPA core only.

| Length | Affine Triton hybrid ms | Attention core ms | Attention / affine |
|---:|---:|---:|---:|
| 8192 | 3.12 | 1.48 | 0.47x |
| 16384 | 5.36 | 6.39 | 1.19x |
| 32768 | 5.48 | 27.32 | 4.98x |

## Systems Snapshot: Backbone Forward Timing

Backbone hidden-state forward only: embeddings, sequence blocks, and
final normalization. This excludes LM head, loss, backward pass,
optimizer, and KV-cache decode behavior.

| Length | MOGT ms | Transformer NoPE ms | Transformer / MOGT |
|---:|---:|---:|---:|
| 8192 | 14.07 | 10.81 | 0.77x |
| 16384 | 37.88 | 35.29 | 0.93x |
| 32768 | 77.56 | 97.05 | 1.25x |

## Language Modeling Pilot: Hybrid Wiring Sanity

This is a 10-step, one-seed WikiText-103 pilot at context 8192,
`d_model=128`, and two layers. It is only a wiring and optimization
sanity check, not a language-modeling quality claim.

| Run | Model | Val loss | PPL | Peak MB | Elapsed s |
|---|---|---:|---:|---:|---:|
| `hybrid_alt_ctx8192_d128_l2_steps10_seed42` | hybrid_alternating_r16_triton_hybrid_cayley_damp0.999 | 10.4828 | 35694.57 | 2502.6 | 3.68 |
| `transformer_ctx8192_d128_l2_steps10_seed42` | scratch_transformer | 10.5214 | 37099.21 | 2502.6 | 2.15 |

## Hybrid Ratio Pilot

These are context-8192, `d_model=128`, four-layer pilot runs.
The 5-step rows are one-seed queueing signals. The 50-step
aggregate uses seeds 7/42/123 and two validation batches.

### 5-Step Ratio Probe

| Steps | MOGT fraction | MOGT layers | Zero-attn init | Val loss | PPL | Elapsed s |
|---:|---:|---:|---|---:|---:|---:|
| 5 | 0.00 | 0 | no | 10.6355 | 41586.72 | 1.14 |
| 5 | 0.25 | 1 | no | 10.6159 | 40777.07 | 2.46 |
| 5 | 0.50 | 2 | no | 10.6100 | 40539.46 | 2.59 |
| 5 | 0.75 | 3 | no | 10.5838 | 39488.74 | 2.68 |
| 5 | 1.00 | 4 | no | 10.5526 | 38275.84 | 2.76 |
| 5 | 0.00 | 0 | yes | 10.6032 | 40264.20 | 1.11 |
| 5 | 0.25 | 1 | yes | 10.5540 | 38331.36 | 2.45 |
| 5 | 0.50 | 2 | yes | 10.5543 | 38343.65 | 2.55 |
| 5 | 0.75 | 3 | yes | 10.5532 | 38298.62 | 2.66 |
| 5 | 1.00 | 4 | yes | 10.5526 | 38275.84 | 2.78 |

### 50-Step Zero-Init Aggregate

| MOGT fraction | Seeds | Mean val loss | Std | Mean PPL | Mean elapsed s |
|---:|---|---:|---:|---:|---:|
| 0.00 | 7,42,123 | 9.5865 | 0.0091 | 14567.81 | 6.14 |
| 0.25 | 7,42,123 | 9.5784 | 0.0163 | 14451.27 | 8.34 |
| 0.50 | 7,42,123 | 9.6034 | 0.0167 | 14816.49 | 9.36 |

Seed-42 100% MOGT zero-init control at 50 steps: val loss 9.6249, PPL 15136.36.

### 200-Step Zero-Init Aggregate

| MOGT fraction | Seeds | Mean val loss | Std | Mean PPL | Mean elapsed s |
|---:|---|---:|---:|---:|---:|
| 0.00 | 7,42,123 | 7.6069 | 0.0091 | 2012.15 | 22.67 |
| 0.25 | 7,42,123 | 7.4897 | 0.0107 | 1789.60 | 27.87 |

### 200-Step Single-Layer Position Ablation

Same setup as the 200-step ratio follow-up. The layer-1 row is the
original 25% ratio run; layers 2 and 3 are explicit-index follow-ups.
Layer 0 currently has only seed 42, so its row is a provisional
diagnostic rather than a full aggregate.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL |
|---|---|---:|---:|---:|---:|
| attention-only | 7,42,123 | 7.6069 | 0.0091 | 0.0000 | 2012.15 |
| layer 0 | 42 | 7.5189 | 0.0000 | -0.0880 | 1842.63 |
| layer 1 | 7,42,123 | 7.4897 | 0.0107 | -0.1172 | 1789.60 |
| layer 2 | 7,42,123 | 7.4675 | 0.0110 | -0.1395 | 1750.22 |
| layer 3 | 7,42,123 | 7.4539 | 0.0092 | -0.1531 | 1726.60 |

### 500-Step Late-Layer Scale-Up

Same context/model size as above, but 500 optimizer steps and eight
validation batches. This targets the best 200-step position rather
than sweeping every layer.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.8137 | 0.0139 | 0.0000 | 910.28 | 57.90 |
| layer 3 | 7,42,123 | 6.7241 | 0.0132 | -0.0896 | 832.29 | 68.92 |

### 1000-Step Late-Layer Scale-Up

Same context/model size as above, but 1000 optimizer steps and eight
validation batches. This targets the best 200-step position rather
than sweeping every layer.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.3868 | 0.0098 | 0.0000 | 593.99 | 114.63 |
| layer 3 | 7,42,123 | 6.3357 | 0.0128 | -0.0511 | 564.42 | 135.77 |

### d_model=192 Width Scale Probe (500 Steps)

Context 8192, four layers, 500 optimizer steps, eight validation
batches, and the same late-layer target as the d_model=128 runs.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.5024 | 0.0141 | 0.0000 | 666.78 | 73.80 | 11420544 |
| layer 3 | 7,42,123 | 6.4544 | 0.0134 | -0.0480 | 635.51 | 82.27 | 11396160 |

### d_model=192 Width Scale Probe (1000 Steps)

Context 8192, four layers, 1000 optimizer steps, eight validation
batches, and the same late-layer target as the d_model=128 runs.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.0997 | 0.0126 | 0.0000 | 445.76 | 147.63 | 11420544 |
| layer 2 | 7,42,123 | 6.1049 | 0.0147 | 0.0052 | 448.09 | 162.59 | 11396160 |
| layer 3 | 7,42,123 | 6.1337 | 0.0162 | 0.0340 | 461.20 | 162.42 | 11396160 |
| layers 1+2 | 7,42,123 | 6.1416 | 0.0180 | 0.0419 | 464.86 | 181.86 | 11371776 |

### d_model=192 Learning-Rate Probe (lr=5e-4)

Context 8192, four layers, 1000 optimizer steps, eight validation
batches. This is a fairness check after lr=5e-4 improved the
layer-2 hybrid seed-42 diagnostic.

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 5.8753 | 0.0108 | 0.0000 | 356.14 | 148.01 | 11420544 |
| layer 2 | 7,42,123 | 5.9058 | 0.0430 | 0.0306 | 367.41 | 163.13 | 11396160 |

### d_model=192 Residual-Scale 0.5 Confirmation

Context 8192, four layers, 1000 optimizer steps, lr=5e-4,
and eight validation batches. This compares the tuned
attention-only control against the fixed residual-scale 0.5
layer-2 hybrid on the same three seeds.

| Position | Residual scale | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| attention-only | 1.00 | 7,42,123 | 5.8753 | 0.0108 | 0.0000 | 356.14 | 148.01 | 11420544 |
| layer 2 | 0.50 | 7,42,123 | 5.8775 | 0.0293 | 0.0022 | 357.01 | 161.72 | 11396160 |

Paired-seed diagnostic: fixed scale 0.5 beats attention on 2/3 seeds (7,123), but its aggregate mean is still the claim boundary.

### d_model=192 MOGT LR Multiplier 0.5 Confirmation

Context 8192, four layers, 1000 optimizer steps, lr=5e-4,
fixed residual scale 0.5, and eight validation batches. This
compares tuned attention-only against the layer-2 hybrid with
a 0.5 learning-rate multiplier on MOGT block parameters.

| Position | Residual scale | MOGT LR mult | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| attention-only | 1.00 | 1.00 | 7,42,123 | 5.8753 | 0.0108 | 0.0000 | 356.14 | 148.01 | 11420544 |
| layer 2 | 0.50 | 0.50 | 7,42,123 | 5.8511 | 0.0117 | -0.0242 | 347.63 | 162.11 | 11396160 |

Paired-seed diagnostic: layer-2 MOGT with residual scale 0.5 and MOGT LR multiplier 0.5 beats attention on 3/3 paired seeds (7,42,123).

### d_model=256 Width Migration Diagnostic (seed 42)

Context 8192, four layers, 1000 optimizer steps, lr=5e-4,
and eight validation batches. This tests whether the d_model=192
MOGT LR multiplier recipe immediately transfers to a wider model.

| Position | Residual scale | MOGT LR mult | Val loss | Delta vs attention seed42 | PPL | Params | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 1.00 | 1.00 | 5.7550 | 0.0000 | 315.76 | 16275968 | `hybrid_scale_d256_l4_steps1000_attn_lr5e4_ctx8192_seed42` |
| layer 2 | 0.50 | 0.50 | 5.7558 | 0.0009 | 316.03 | 16210688 | `hybrid_scale_d256_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42` |

### d_model=192 Residual-Scale Sweep (seed 42)

Single-seed diagnostic at context 8192, four layers, 1000 optimizer
steps, lr=5e-4, and eight validation batches. This table is not
mixed into the aggregate above.

| Position | Residual scale | Val loss | Delta vs attention seed42 | PPL | Run |
|---|---:|---:|---:|---:|---|
| attention-only | 1.00 | 5.8877 | 0.0000 | 360.57 | `hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42` |
| layer 2 | 0.25 | 5.9199 | 0.0322 | 372.37 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p25_ctx8192_seed42` |
| layer 2 | 0.50 | 5.9110 | 0.0233 | 369.07 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42` |
| layer 2 | 0.75 | 5.9174 | 0.0297 | 371.44 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p75_ctx8192_seed42` |
| layer 2 | 1.00 | 5.9545 | 0.0669 | 385.50 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_ctx8192_seed42` |

### d_model=192 Residual-Scale Schedule Diagnostic (seed 42)

Single-seed diagnostic at context 8192, four layers, 1000 optimizer
steps, lr=5e-4, and eight validation batches. Scheduled runs are
reported separately from fixed-scale aggregates.

| Position | Schedule | Scale path | Val loss | Delta vs attention seed42 | PPL | Run |
|---|---|---|---:|---:|---:|---|
| attention-only | constant | 1.00 | 5.8877 | 0.0000 | 360.57 | `hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42` |
| layer 2 | constant | 0.50 | 5.9110 | 0.0233 | 369.07 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42` |
| layer 2 | linear_warmup | 0.25 -> 0.50 / 250 steps | 5.9497 | 0.0620 | 383.65 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtsched0p25to0p5s250_ctx8192_seed42` |

### d_model=192 MOGT Learning-Rate Multiplier Diagnostic (seed 42)

Single-seed diagnostic at context 8192, four layers, 1000 optimizer
steps, lr=5e-4, fixed residual scale 0.5, and eight validation
batches. Non-default MOGT optimizer groups are reported separately
from the fixed-scale aggregate.

| Position | Residual scale | MOGT LR mult | Val loss | Delta vs attention seed42 | PPL | Run |
|---|---:|---:|---:|---:|---:|---|
| attention-only | 1.00 | 1.00 | 5.8877 | 0.0000 | 360.57 | `hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42` |
| layer 2 | 0.50 | 0.50 | 5.8601 | -0.0276 | 350.75 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42` |
| layer 2 | 0.50 | 1.00 | 5.9110 | 0.0233 | 369.07 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42` |

### d_model=192 Learned Residual-Gate Diagnostic (seed 42)

Single-seed diagnostic at context 8192, four layers, 1000 optimizer
steps, lr=5e-4, and eight validation batches. The learned gate is
reported separately because it changes the optimization dynamics.

| Position | Gate | Init / fixed scale | Val loss | Delta vs attention seed42 | PPL | Run |
|---|---|---:|---:|---:|---:|---|
| layer 2 | fixed | 0.50 | 5.9110 | 0.0233 | 369.07 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42` |
| layer 2 | learned | 0.50 | 5.9752 | 0.0875 | 393.54 | `hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtgate0p5_ctx8192_seed42` |

## Claim Boundary

Safe current wording:

> Coupling token-dependent writing with forgetting gives a scan-compatible
> matrix-valued recurrent operator that solves overwrite state tracking.
> Adding prefix-conditioned slot addressing extends this mechanism to tracked
> multi-slot routing under controlled synthetic settings.

Unsafe current wording:

> MOGT generally beats Transformer, solves language modeling, or replaces
> attention across tasks.
