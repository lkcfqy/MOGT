# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=128`, 4 layers, 200 optimizer steps,
WikiText-103/GPT-2 token stream.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 7,42,123 | 7.6069 | 0.0091 | 0.0000 | 2012.15 | 22.67 |
| layer-0 | 42 | 7.5189 | 0.0000 | -0.0880 | 1842.63 | 27.52 |
| layer-1 | 7,42,123 | 7.4897 | 0.0107 | -0.1172 | 1789.60 | 27.87 |
| layer-2 | 7,42,123 | 7.4675 | 0.0110 | -0.1395 | 1750.22 | 28.80 |
| layer-3 | 7,42,123 | 7.4539 | 0.0092 | -0.1531 | 1726.60 | 28.39 |

Layer 0 currently has only one seed in the 200-step table; layer 1 is the original 25% ratio run, and layers 2/3 are explicit-index follow-ups.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 7 | 7.6174 | 2033.28 | 7.5554 | 2516.6 | 22.40 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed7_steps200.json` |
| attention-only | 42 | 7.6006 | 1999.44 | 7.6702 | 2516.6 | 22.79 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed42_steps200.json` |
| attention-only | 123 | 7.6028 | 2003.74 | 7.5242 | 2516.6 | 22.83 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed123_steps200.json` |
| layer-0 | 42 | 7.5189 | 1842.63 | 7.5881 | 2516.6 | 27.52 | `benchmark_runs/hybrid_layeridx_steps200_l0_ctx8192_d128_l4_seed42.json` |
| layer-1 | 7 | 7.4968 | 1802.30 | 7.4307 | 2516.6 | 27.54 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed7_steps200.json` |
| layer-1 | 42 | 7.4774 | 1767.56 | 7.5521 | 2516.6 | 28.03 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed42_steps200.json` |
| layer-1 | 123 | 7.4950 | 1798.94 | 7.3989 | 2516.6 | 28.03 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed123_steps200.json` |
| layer-2 | 7 | 7.4795 | 1771.30 | 7.4120 | 2516.6 | 30.21 | `benchmark_runs/hybrid_layeridx_steps200_l2_ctx8192_d128_l4_seed7.json` |
| layer-2 | 42 | 7.4578 | 1733.33 | 7.5355 | 2516.6 | 28.09 | `benchmark_runs/hybrid_layeridx_steps200_l2_ctx8192_d128_l4_seed42.json` |
| layer-2 | 123 | 7.4651 | 1746.05 | 7.3662 | 2516.6 | 28.11 | `benchmark_runs/hybrid_layeridx_steps200_l2_ctx8192_d128_l4_seed123.json` |
| layer-3 | 7 | 7.4642 | 1744.41 | 7.3972 | 2516.6 | 28.22 | `benchmark_runs/hybrid_layeridx_steps200_l3_ctx8192_d128_l4_seed7.json` |
| layer-3 | 42 | 7.4464 | 1713.65 | 7.5224 | 2516.6 | 28.43 | `benchmark_runs/hybrid_layeridx_steps200_l3_ctx8192_d128_l4_seed42.json` |
| layer-3 | 123 | 7.4511 | 1721.74 | 7.3495 | 2516.6 | 28.53 | `benchmark_runs/hybrid_layeridx_steps200_l3_ctx8192_d128_l4_seed123.json` |
