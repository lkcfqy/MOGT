# Hybrid LM Ratio Sweep Summary

Generated from `mogt-experiment-v1` language-modeling reports.
These are pilot runs unless the command manifest says otherwise.

## Aggregate

| Family | Fraction | Zero-attn init | Seeds | Steps | Mean val loss | Std | Mean PPL | Mean elapsed s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `hybrid_ratio_sweep_pilot` | 0.00 | no | 42 | 5 | 10.6355 | 0.0000 | 41586.72 | 1.14 |
| `hybrid_ratio_sweep_pilot` | 0.25 | no | 42 | 5 | 10.6159 | 0.0000 | 40777.07 | 2.46 |
| `hybrid_ratio_sweep_pilot` | 0.50 | no | 42 | 5 | 10.6100 | 0.0000 | 40539.46 | 2.59 |
| `hybrid_ratio_sweep_pilot` | 0.75 | no | 42 | 5 | 10.5838 | 0.0000 | 39488.74 | 2.68 |
| `hybrid_ratio_sweep_pilot` | 1.00 | no | 42 | 5 | 10.5526 | 0.0000 | 38275.84 | 2.76 |
| `hybrid_ratio_sweep_zeroattn_pilot` | 0.00 | yes | 42 | 5 | 10.6032 | 0.0000 | 40264.20 | 1.11 |
| `hybrid_ratio_sweep_zeroattn_pilot` | 0.25 | yes | 42 | 5 | 10.5540 | 0.0000 | 38331.36 | 2.45 |
| `hybrid_ratio_sweep_zeroattn_pilot` | 0.50 | yes | 42 | 5 | 10.5543 | 0.0000 | 38343.65 | 2.55 |
| `hybrid_ratio_sweep_zeroattn_pilot` | 0.75 | yes | 42 | 5 | 10.5532 | 0.0000 | 38298.62 | 2.66 |
| `hybrid_ratio_sweep_zeroattn_pilot` | 1.00 | yes | 42 | 5 | 10.5526 | 0.0000 | 38275.84 | 2.78 |
| `hybrid_ratio_sweep_zeroattn_steps200` | 0.00 | yes | 7,42,123 | 200 | 7.6069 | 0.0091 | 2012.15 | 22.67 |
| `hybrid_ratio_sweep_zeroattn_steps200` | 0.25 | yes | 7,42,123 | 200 | 7.4897 | 0.0107 | 1789.60 | 27.87 |
| `hybrid_ratio_sweep_zeroattn_steps50` | 0.00 | yes | 42 | 50 | 9.5796 | 0.0000 | 14466.43 | 6.09 |
| `hybrid_ratio_sweep_zeroattn_steps50` | 0.25 | yes | 42 | 50 | 9.5618 | 0.0000 | 14211.05 | 8.28 |
| `hybrid_ratio_sweep_zeroattn_steps50` | 1.00 | yes | 42 | 50 | 9.6249 | 0.0000 | 15136.36 | 11.22 |
| `hybrid_ratio_sweep_zeroattn_steps50_3seed` | 0.00 | yes | 7,42,123 | 50 | 9.5865 | 0.0091 | 14567.81 | 6.14 |
| `hybrid_ratio_sweep_zeroattn_steps50_3seed` | 0.25 | yes | 7,42,123 | 50 | 9.5784 | 0.0163 | 14451.27 | 8.34 |
| `hybrid_ratio_sweep_zeroattn_steps50_3seed` | 0.50 | yes | 7,42,123 | 50 | 9.6034 | 0.0167 | 14816.49 | 9.36 |

## Runs

| Fraction | MOGT layers | Zero-attn init | Seed | Steps | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0 | no | 42 | 5 | 10.6355 | 41586.72 | 10.6357 | 2516.6 | 1.14 | `benchmark_runs/hybrid_ratio_sweep_pilot_frac0_ctx8192_d128_l4_seed42_steps5.json` |
| 0.25 | 1 | no | 42 | 5 | 10.6159 | 40777.07 | 10.6151 | 2516.6 | 2.46 | `benchmark_runs/hybrid_ratio_sweep_pilot_frac0p25_ctx8192_d128_l4_seed42_steps5.json` |
| 0.50 | 2 | no | 42 | 5 | 10.6100 | 40539.46 | 10.6110 | 2516.6 | 2.59 | `benchmark_runs/hybrid_ratio_sweep_pilot_frac0p5_ctx8192_d128_l4_seed42_steps5.json` |
| 0.75 | 3 | no | 42 | 5 | 10.5838 | 39488.74 | 10.5781 | 2516.6 | 2.68 | `benchmark_runs/hybrid_ratio_sweep_pilot_frac0p75_ctx8192_d128_l4_seed42_steps5.json` |
| 1.00 | 4 | no | 42 | 5 | 10.5526 | 38275.84 | 10.5402 | 2516.6 | 2.76 | `benchmark_runs/hybrid_ratio_sweep_pilot_frac1_ctx8192_d128_l4_seed42_steps5.json` |
| 0.00 | 0 | yes | 42 | 5 | 10.6032 | 40264.20 | 10.6086 | 2516.6 | 1.11 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_pilot_frac0_ctx8192_d128_l4_seed42_steps5.json` |
| 0.25 | 1 | yes | 42 | 5 | 10.5540 | 38331.36 | 10.5426 | 2516.6 | 2.45 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_pilot_frac0p25_ctx8192_d128_l4_seed42_steps5.json` |
| 0.50 | 2 | yes | 42 | 5 | 10.5543 | 38343.65 | 10.5408 | 2516.6 | 2.55 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_pilot_frac0p5_ctx8192_d128_l4_seed42_steps5.json` |
| 0.75 | 3 | yes | 42 | 5 | 10.5532 | 38298.62 | 10.5423 | 2516.6 | 2.66 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_pilot_frac0p75_ctx8192_d128_l4_seed42_steps5.json` |
| 1.00 | 4 | yes | 42 | 5 | 10.5526 | 38275.84 | 10.5402 | 2516.6 | 2.78 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_pilot_frac1_ctx8192_d128_l4_seed42_steps5.json` |
| 0.00 | 0 | yes | 7 | 50 | 9.5968 | 14717.61 | 9.5932 | 2516.6 | 6.11 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0_ctx8192_d128_l4_seed7_steps50.json` |
| 0.25 | 1 | yes | 7 | 50 | 9.5926 | 14655.70 | 9.5819 | 2516.6 | 8.31 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p25_ctx8192_d128_l4_seed7_steps50.json` |
| 0.50 | 2 | yes | 7 | 50 | 9.6188 | 15045.46 | 9.6176 | 2516.6 | 9.29 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p5_ctx8192_d128_l4_seed7_steps50.json` |
| 0.00 | 0 | yes | 42 | 50 | 9.5796 | 14466.18 | 9.5733 | 2516.6 | 6.09 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0_ctx8192_d128_l4_seed42_steps50.json` |
| 0.00 | 0 | yes | 42 | 50 | 9.5796 | 14466.43 | 9.5733 | 2516.6 | 6.09 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_frac0_ctx8192_d128_l4_seed42_steps50.json` |
| 0.25 | 1 | yes | 42 | 50 | 9.5606 | 14193.96 | 9.5555 | 2516.6 | 8.33 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p25_ctx8192_d128_l4_seed42_steps50.json` |
| 0.25 | 1 | yes | 42 | 50 | 9.5618 | 14211.05 | 9.5572 | 2516.6 | 8.28 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_frac0p25_ctx8192_d128_l4_seed42_steps50.json` |
| 0.50 | 2 | yes | 42 | 50 | 9.5857 | 14555.13 | 9.5839 | 2516.6 | 9.34 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p5_ctx8192_d128_l4_seed42_steps50.json` |
| 1.00 | 4 | yes | 42 | 50 | 9.6249 | 15136.36 | 9.6282 | 2516.6 | 11.22 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_frac1_ctx8192_d128_l4_seed42_steps50.json` |
| 0.00 | 0 | yes | 123 | 50 | 9.5833 | 14519.64 | 9.6322 | 2516.6 | 6.23 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0_ctx8192_d128_l4_seed123_steps50.json` |
| 0.25 | 1 | yes | 123 | 50 | 9.5822 | 14504.14 | 9.6340 | 2516.6 | 8.38 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p25_ctx8192_d128_l4_seed123_steps50.json` |
| 0.50 | 2 | yes | 123 | 50 | 9.6057 | 14848.86 | 9.6548 | 2516.6 | 9.44 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps50_3seed_frac0p5_ctx8192_d128_l4_seed123_steps50.json` |
| 0.00 | 0 | yes | 7 | 200 | 7.6174 | 2033.28 | 7.5554 | 2516.6 | 22.40 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed7_steps200.json` |
| 0.25 | 1 | yes | 7 | 200 | 7.4968 | 1802.30 | 7.4307 | 2516.6 | 27.54 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed7_steps200.json` |
| 0.00 | 0 | yes | 42 | 200 | 7.6006 | 1999.44 | 7.6702 | 2516.6 | 22.79 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed42_steps200.json` |
| 0.25 | 1 | yes | 42 | 200 | 7.4774 | 1767.56 | 7.5521 | 2516.6 | 28.03 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed42_steps200.json` |
| 0.00 | 0 | yes | 123 | 200 | 7.6028 | 2003.74 | 7.5242 | 2516.6 | 22.83 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0_ctx8192_d128_l4_seed123_steps200.json` |
| 0.25 | 1 | yes | 123 | 200 | 7.4950 | 1798.94 | 7.3989 | 2516.6 | 28.03 | `benchmark_runs/hybrid_ratio_sweep_zeroattn_steps200_frac0p25_ctx8192_d128_l4_seed123_steps200.json` |
