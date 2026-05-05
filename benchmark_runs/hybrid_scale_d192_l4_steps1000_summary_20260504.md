# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=192`, 4 layers, 1000 optimizer steps,
WikiText-103/GPT-2 token stream.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 6.0997 | 0.0126 | 0.0000 | 445.76 | 147.63 |
| layer-2 | 7,42,123 | 6.1049 | 0.0147 | 0.0052 | 448.09 | 162.59 |
| layer-3 | 7,42,123 | 6.1337 | 0.0162 | 0.0340 | 461.20 | 162.42 |
| layers-1-2 | 7,42,123 | 6.1416 | 0.0180 | 0.0419 | 464.86 | 181.86 |

This filtered summary contains attention-only, layer-2, layer-3, layers-1-2; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 6.0909 | 441.83 | 6.2933 | 2589.9 | 146.77 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_ctx8192_seed7.json` |
| attention-only | 42 | 6.1142 | 452.24 | 6.0672 | 2589.9 | 148.05 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_ctx8192_seed42.json` |
| attention-only | 123 | 6.0941 | 443.22 | 5.9156 | 2589.9 | 148.09 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_ctx8192_seed123.json` |
| layer-2 | 7 | 6.0921 | 442.33 | 6.2963 | 2589.6 | 162.17 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_ctx8192_seed7.json` |
| layer-2 | 42 | 6.1210 | 455.31 | 6.0733 | 2589.6 | 162.69 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_ctx8192_seed42.json` |
| layer-2 | 123 | 6.1017 | 446.64 | 5.9153 | 2589.6 | 162.91 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_ctx8192_seed123.json` |
| layer-3 | 7 | 6.1319 | 460.30 | 6.3292 | 2589.6 | 162.61 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l3_ctx8192_seed7.json` |
| layer-3 | 42 | 6.1508 | 469.09 | 6.1103 | 2589.6 | 162.33 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l3_ctx8192_seed42.json` |
| layer-3 | 123 | 6.1186 | 454.21 | 5.9304 | 2589.6 | 162.33 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l3_ctx8192_seed123.json` |
| layers-1-2 | 7 | 6.1274 | 458.24 | 6.3441 | 2589.4 | 181.67 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l12_ctx8192_seed7.json` |
| layers-1-2 | 42 | 6.1357 | 462.06 | 6.0935 | 2589.4 | 182.18 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l12_ctx8192_seed42.json` |
| layers-1-2 | 123 | 6.1618 | 474.28 | 5.9912 | 2589.4 | 181.73 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l12_ctx8192_seed123.json` |
