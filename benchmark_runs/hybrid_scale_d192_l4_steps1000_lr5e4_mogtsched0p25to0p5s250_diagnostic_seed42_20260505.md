# d_model=192 Layer-2 Residual-Scale Schedule Diagnostic

Context 8192, four layers, 1000 optimizer steps, lr=5e-4, eight validation
batches, WikiText-103/GPT-2 token stream, seed 42, zero-initialized attention
output projections.

## Result

| Run | Residual scale path | Val loss | PPL | Delta vs attention seed42 |
|---|---|---:|---:|---:|
| attention-only | constant 1.00 | 5.8877 | 360.57 | 0.0000 |
| layer2 fixed scale | constant 0.50 | 5.9110 | 369.07 | +0.0233 |
| layer2 schedule | 0.25 -> 0.50 over 250 steps | 5.9497 | 383.65 | +0.0620 |

## Interpretation

The fixed readout residual scale of 0.5 remains better than this conservative
linear warmup. A simple scale schedule does not repair the seed-42 gap; future
LM work should prioritize richer residual dynamics, auxiliary state-tracking
losses, or a different hybrid placement/training recipe rather than extending
this schedule family immediately.

## Artifacts

- `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtsched0p25to0p5s250_ctx8192_seed42.json`
- `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json`
- `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42.json`
