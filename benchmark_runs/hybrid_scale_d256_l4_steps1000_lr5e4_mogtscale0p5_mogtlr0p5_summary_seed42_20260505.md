# Hybrid LM Single-Layer Position Summary

Context 8192, `d_model=256`, 4 layers, 1000 optimizer steps,
WikiText-103/GPT-2 token stream.
Learning-rate filter: `0.0005`.
MOGT residual-scale filter: `0.5`.
MOGT FFN residual-scale filter: default `1.0`.
MOGT residual-gate filter: default `False`.
MOGT LR multiplier filter: `0.5`.
All rows use `--zero-init-attention-out`.

## Aggregate

| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |
|---|---|---:|---:|---:|---:|---:|
| attention-only | 42 | 5.7550 | 0.0000 | 0.0000 | 315.76 | 163.35 |
| layer-2 | 42 | 5.7558 | 0.0000 | 0.0009 | 316.03 | 178.96 |

This filtered summary contains attention-only, layer-2; absent layers were not part of this run set.
Treat the layer-order trend as a scaling target, not a final LM claim.

## Runs

| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| attention-only | 42 | 5.7550 | 315.76 | 5.7309 | 2674.7 | 163.35 | `benchmark_runs/hybrid_scale_d256_l4_steps1000_attn_lr5e4_ctx8192_seed42.json` |
| layer-2 | 42 | 5.7558 | 316.03 | 5.7252 | 2673.9 | 178.96 | `benchmark_runs/hybrid_scale_d256_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42.json` |
