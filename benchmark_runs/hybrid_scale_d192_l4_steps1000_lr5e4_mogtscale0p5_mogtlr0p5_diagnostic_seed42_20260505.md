# d_model=192 Layer-2 MOGT LR Multiplier Diagnostic

Context 8192, four layers, 1000 optimizer steps, lr=5e-4, eight validation
batches, WikiText-103/GPT-2 token stream, seed 42, zero-initialized attention
output projections.

## Result

| Run | Residual scale | MOGT LR mult | Val loss | PPL | Delta vs attention seed42 |
|---|---:|---:|---:|---:|---:|
| attention-only | 1.00 | 1.00 | 5.8877 | 360.57 | 0.0000 |
| layer2 fixed scale | 0.50 | 1.00 | 5.9110 | 369.07 | +0.0233 |
| layer2 fixed scale + lower MOGT LR | 0.50 | 0.50 | 5.8601 | 350.75 | -0.0276 |

## Interpretation

This is the first d_model=192, 1000-step, lr=5e-4 layer-2 diagnostic where the
hybrid beats the same-seed attention-only control. The result suggests the bad
seed42 behavior was at least partly an optimizer-partition issue: MOGT block
parameters benefit from a lower learning rate than the surrounding Transformer
backbone. This needs seeds 7 and 123 before it can become an aggregate LM
claim.

## Artifacts

- `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42.json`
- `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json`
- `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42.json`
