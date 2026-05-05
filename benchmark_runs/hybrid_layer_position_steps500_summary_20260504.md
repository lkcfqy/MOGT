# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=128`, 4 layers, 500 optimizer steps,
WikiText-103/GPT-2 token stream.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.8137 | 0.0139 | 0.0000 | 910.28 | 57.90 |
| layer-3 | 7,42,123 | 6.7241 | 0.0132 | -0.0896 | 832.29 | 68.92 |

This filtered summary contains attention-only, layer-3; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 6.8075 | 904.62 | 6.7094 | 2516.6 | 56.73 | `benchmark_runs/hybrid_late_l3_steps500_attn_ctx8192_d128_l4_seed7.json` |
| attention-only | 42 | 6.8296 | 924.81 | 6.8591 | 2516.6 | 58.80 | `benchmark_runs/hybrid_late_l3_steps500_attn_ctx8192_d128_l4_seed42.json` |
| attention-only | 123 | 6.8040 | 901.42 | 6.9809 | 2516.6 | 58.16 | `benchmark_runs/hybrid_late_l3_steps500_attn_ctx8192_d128_l4_seed123.json` |
| layer-3 | 7 | 6.7102 | 820.75 | 6.6039 | 2516.6 | 68.93 | `benchmark_runs/hybrid_late_l3_steps500_l3_ctx8192_d128_l4_seed7.json` |
| layer-3 | 42 | 6.7366 | 842.67 | 6.7683 | 2516.6 | 69.08 | `benchmark_runs/hybrid_late_l3_steps500_l3_ctx8192_d128_l4_seed42.json` |
| layer-3 | 123 | 6.7256 | 833.44 | 6.9071 | 2516.6 | 68.74 | `benchmark_runs/hybrid_late_l3_steps500_l3_ctx8192_d128_l4_seed123.json` |
