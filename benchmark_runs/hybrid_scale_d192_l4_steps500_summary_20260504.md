# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=192`, 4 layers, 500 optimizer steps,
WikiText-103/GPT-2 token stream.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.5024 | 0.0141 | 0.0000 | 666.78 | 73.80 |
| layer-3 | 7,42,123 | 6.4544 | 0.0134 | -0.0480 | 635.51 | 82.27 |

This filtered summary contains attention-only, layer-3; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 6.4997 | 664.93 | 6.4057 | 2589.9 | 72.98 | `benchmark_runs/hybrid_scale_d192_l4_steps500_attn_ctx8192_seed7.json` |
| attention-only | 42 | 6.5177 | 677.02 | 6.5453 | 2589.9 | 74.14 | `benchmark_runs/hybrid_scale_d192_l4_steps500_attn_ctx8192_seed42.json` |
| attention-only | 123 | 6.4898 | 658.40 | 6.6957 | 2589.9 | 74.28 | `benchmark_runs/hybrid_scale_d192_l4_steps500_attn_ctx8192_seed123.json` |
| layer-3 | 7 | 6.4454 | 629.82 | 6.3483 | 2589.6 | 82.15 | `benchmark_runs/hybrid_scale_d192_l4_steps500_l3_ctx8192_seed7.json` |
| layer-3 | 42 | 6.4697 | 645.31 | 6.4704 | 2589.6 | 82.35 | `benchmark_runs/hybrid_scale_d192_l4_steps500_l3_ctx8192_seed42.json` |
| layer-3 | 123 | 6.4479 | 631.40 | 6.6515 | 2589.6 | 82.31 | `benchmark_runs/hybrid_scale_d192_l4_steps500_l3_ctx8192_seed123.json` |
