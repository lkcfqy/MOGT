# Long-Context Probe Summary

Protocol: WikiText-103 raw v1, GPT-2 token stream, `ctx=65536`, `batch_size=1`, `grad_accum_steps=1`, `max_steps=1`, L4 24GB.

| Model | Config | Status | Loss chunk | Train loss | Peak MB | Evidence |
|---|---|---|---:|---:|---:|---|
| MOGT baseline_v1 shape | 12L d768 r16, `triton_hybrid`, `cayley` | OOM | 256 | - | ~22500 | `train_mogt_probe_ctx65536_seed42_20260503.log` |
| MOGT baseline_v1 shape | 12L d768 r16, `triton_hybrid`, `cayley` | OOM | 64 | - | ~22500 | `train_mogt_probe_ctx65536_losschunk64_seed42_20260503.log` |
| MOGT baseline_v1 shape | 12L d768 r16, `triton_hybrid`, `cayley` | OOM | 16 | - | ~22500 | `train_mogt_probe_ctx65536_losschunk16_seed42_20260503.log` |
| MOGT baseline_v1 shape | 12L d768 r16, `triton_hybrid`, `cayley` | OOM | 1 | - | ~22500 | `train_mogt_probe_ctx65536_losschunk1_seed42_20260503.log` |
| MOGT + memory-efficient LM loss | 12L d768 r16, `triton_hybrid`, `cayley` | OK | 256 | 15.2754 | - | `train_mogt_probe_ctx65536_memloss_seed42_20260503.log` |
| MOGT + memory-efficient LM loss | 12L d768 r16, `triton_hybrid`, `cayley` | OK | 256 | 15.3442 | 6561 | no-checkpoint random-token probe |
| Scratch Transformer + memory-efficient LM loss | 12L d768 h12, RoPE, SDPA | OK | 256 | 10.9766 | 6147 | no-checkpoint random-token probe |
| Scratch Transformer + memory-efficient LM loss | 12L d768 h12, RoPE, SDPA | OK | 256 | 11.0382 | 6147 | `benchmark_runs/transformer_scratch_probe_ctx65536_seed42.json` |

Interpretation: this is a systems probe, not a quality benchmark. The original retained-logits chunked loss prevented MOGT 130M from fitting `ctx=65536` on L4 24GB. After replacing it with a recompute-in-backward linear CE, MOGT fits a 64k train step. Transformer also fits with the same loss path, so this supports a memory fix and a possible MOGT step-time angle, not a modeling-quality win.
