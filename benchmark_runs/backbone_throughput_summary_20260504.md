# Backbone Throughput Summary

Source: `benchmark_runs/backbone_throughput_identity_coupled_d768_l2_len8192_16384_32768_20260504.json`

Backbone hidden-state forward only: embeddings, sequence blocks, and final
normalization. This excludes LM head, loss, backward pass, optimizer, and
KV-cache decode behavior.

Device: `NVIDIA L4`. `d_model=768`, `num_layers=2`, batch size 1.

| Length | MOGT ms | Transformer NoPE ms | Transformer / MOGT | MOGT peak MB | Transformer peak MB |
|---:|---:|---:|---:|---:|---:|
| 8192 | 14.07 | 10.81 | 0.77x | 445.88 | 268.83 |
| 16384 | 37.88 | 35.29 | 0.93x | 796.07 | 435.51 |
| 32768 | 77.56 | 97.05 | 1.25x | 1503.07 | 771.64 |

Interpretation:

- At 8k and 16k, the small NoPE Transformer backbone remains competitive.
- At 32k, the identity coupled MOGT backbone is faster in this measurement.
- This is still not an end-to-end training or generation throughput claim.
