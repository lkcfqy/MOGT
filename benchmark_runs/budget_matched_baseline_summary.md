# Budget-Matched Baseline Summary

| Model | Context | Params | Config | Seeds | Steps | Status | Val batches | Best loss | Best PPL | Peak MB | Elapsed | Notes |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| MOGT baseline_v1 Cayley | 32768 | - | 12L d768 r16, triton_hybrid, cayley | 3 | 200 | ok | 5/5/5 | 6.4067 | 607.45 | - | - | Checkpoint-only eval of three 200-step seeds. |
| Scratch Mamba SSM | 32768 | 129.12M | 24L d768 | 3 | 200 | ok | 5/5/5 | 9.6168 | 15059.19 | - | - | Aggregate scratch baseline report. |
| Scratch Transformer | 32768 | 123.55M | 12L d768 h12 | 1 | 200 | ok | 5 | 6.1009 | 446.24 | 10103 | 5959.1s | Single-seed scratch report. |

Notes:
- MOGT values come from checkpoint-only evaluation of the three baseline_v1 seeds.
- Scratch baselines use the repo GPT-2 token stream and WikiText-103 data protocol.
- OOM rows should be kept as long-context systems evidence rather than omitted.
