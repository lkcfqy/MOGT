# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=192`, 4 layers, 1000 optimizer steps,
WikiText-103/GPT-2 token stream.
Learning-rate filter: `0.0005`.
MOGT residual-scale filter: `0.5`.
MOGT FFN residual-scale filter: default `1.0`.
MOGT residual-gate filter: default `False`.
MOGT LR multiplier filter: default `1.0`.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 5.8753 | 0.0108 | 0.0000 | 356.14 | 148.01 |
| layer-2 | 7,42,123 | 5.8775 | 0.0293 | 0.0022 | 357.01 | 161.72 |

This filtered summary contains attention-only, layer-2; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 5.8698 | 354.19 | 6.1033 | 2589.9 | 148.05 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed7.json` |
| attention-only | 42 | 5.8877 | 360.57 | 5.8587 | 2589.9 | 148.04 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42.json` |
| attention-only | 123 | 5.8683 | 353.65 | 5.6440 | 2589.9 | 147.93 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed123.json` |
| layer-2 | 7 | 5.8646 | 352.34 | 6.0970 | 2589.6 | 161.31 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed7.json` |
| layer-2 | 42 | 5.9110 | 369.07 | 5.8690 | 2589.6 | 161.06 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json` |
| layer-2 | 123 | 5.8569 | 349.63 | 5.6380 | 2589.6 | 162.77 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed123.json` |
