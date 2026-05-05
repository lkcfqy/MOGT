# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=128`, 4 layers, 1000 optimizer steps,
WikiText-103/GPT-2 token stream.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.3868 | 0.0098 | 0.0000 | 593.99 | 114.63 |
| layer-3 | 7,42,123 | 6.3357 | 0.0128 | -0.0511 | 564.42 | 135.77 |

This filtered summary contains attention-only, layer-3; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 6.3956 | 599.21 | 6.5769 | 2516.6 | 113.88 | `benchmark_runs/hybrid_late_l3_steps1000_attn_ctx8192_d128_l4_seed7.json` |
| attention-only | 42 | 6.3763 | 587.75 | 6.3305 | 2516.6 | 114.88 | `benchmark_runs/hybrid_late_l3_steps1000_attn_ctx8192_d128_l4_seed42.json` |
| attention-only | 123 | 6.3886 | 595.02 | 6.2562 | 2516.6 | 115.13 | `benchmark_runs/hybrid_late_l3_steps1000_attn_ctx8192_d128_l4_seed123.json` |
| layer-3 | 7 | 6.3297 | 560.98 | 6.5070 | 2516.6 | 135.66 | `benchmark_runs/hybrid_late_l3_steps1000_l3_ctx8192_d128_l4_seed7.json` |
| layer-3 | 42 | 6.3505 | 572.77 | 6.3090 | 2516.6 | 135.28 | `benchmark_runs/hybrid_late_l3_steps1000_l3_ctx8192_d128_l4_seed42.json` |
| layer-3 | 123 | 6.3270 | 559.49 | 6.1828 | 2516.6 | 136.36 | `benchmark_runs/hybrid_late_l3_steps1000_l3_ctx8192_d128_l4_seed123.json` |
