# Hybrid LM Residual-Scale Sweep

Context 8192, `d_model=192`, 4 layers, 1000 optimizer steps, lr `5e-4`,
WikiText-103/GPT-2 token stream, seed 42. All MOGT rows use a single layer-2
insertion, `--zero-init-attention-out`, and `--mogt-ffn-residual-scale 1.0`.

This is a single-seed diagnostic, not an aggregate LM result.

| Model / position | MOGT residual scale | Val loss | Delta vs attention | PPL | Artifact |
|---|---:|---:|---:|---:|---|
| attention-only | - | 5.8877 | 0.0000 | 360.57 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42.json` |
| layer-2 MOGT | 0.25 | 5.9199 | 0.0322 | 372.37 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p25_ctx8192_seed42.json` |
| layer-2 MOGT | 0.50 | 5.9110 | 0.0233 | 369.07 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json` |
| layer-2 MOGT | 0.75 | 5.9174 | 0.0297 | 371.44 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p75_ctx8192_seed42.json` |
| layer-2 MOGT | 1.00 | 5.9545 | 0.0669 | 385.50 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_ctx8192_seed42.json` |

Interpretation: fixed residual scaling materially narrows the d192/lr5e-4
layer-2 gap, with the best observed point at scale 0.5. The tuned attention
control still wins at the same seed, so this supports learned or scheduled
residual mixing as the next architecture change rather than a language-modeling
superiority claim.
