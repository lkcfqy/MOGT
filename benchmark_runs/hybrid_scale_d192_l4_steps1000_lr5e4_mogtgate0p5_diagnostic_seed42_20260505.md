# Hybrid LM Learned Residual-Gate Diagnostic

Context 8192, `d_model=192`, 4 layers, 1000 optimizer steps, lr `5e-4`,
WikiText-103/GPT-2 token stream, seed 42. All MOGT rows use a single layer-2
insertion and `--zero-init-attention-out`.

This is a single-seed diagnostic, not an aggregate LM result.

| Model / position | Fusion mode | Init / fixed scale | Val loss | Delta vs attention | PPL | Artifact |
|---|---|---:|---:|---:|---:|---|
| attention-only | - | - | 5.8877 | 0.0000 | 360.57 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_attn_lr5e4_ctx8192_seed42.json` |
| layer-2 MOGT | fixed scale | 0.50 | 5.9110 | 0.0233 | 369.07 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json` |
| layer-2 MOGT | learned gate | 0.50 | 5.9752 | 0.0875 | 393.54 | `benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtgate0p5_ctx8192_seed42.json` |

Interpretation: the naive learned scalar gate is a negative result in this
setup. It starts from the best fixed scale but trains worse, suggesting that
gate dynamics need constraints such as lower learning rate, regularization
toward the initialization, delayed unfreezing, or a bounded residual schedule.
