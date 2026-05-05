# Core Operator Throughput Summary

Source: `benchmark_runs/throughput_core_operator_d768_len8192_16384_32768_20260504.json`

This is a core-operator timing summary, not an end-to-end model throughput
claim. Affine scan timings exclude connection/value projection, matrix
exponential/Cayley construction, normalization, FFN, and LM head. Attention
timings measure FlashAttention/SDPA core only.

Device: `NVIDIA L4`. Batch size: 1. `d_model=768`, `rank=16`. Warmup/iters: 3/10.

| Length | Affine Triton hybrid ms | Attention core ms | Attention / affine | Transport-only ms | Parallel ref ms |
|---:|---:|---:|---:|---:|---:|
| 8192 | 3.12 | 1.48 | 0.47x | 1.10 | 12.81 |
| 16384 | 5.36 | 6.39 | 1.19x | 4.25 | 34.05 |
| 32768 | 5.48 | 27.32 | 4.98x | 9.03 | 75.88 |

Interpretation:

- At 8k, attention core remains faster in this measurement.
- At 16k and 32k, the affine scan core is faster than attention core.
- This supports a systems hypothesis, not a full-model speed claim.
- The next systems step is to profile full MOGT blocks and fused/near-fused
  connection + scan + readout paths.
