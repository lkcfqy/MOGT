# Budget-Matched Scratch Baseline

Source: `benchmark_runs/mamba_scratch_budget_v1_ctx32768_multiseed_20260429.json`

This table is a training-budget-matched baseline, not a pretrained quality anchor. The runs use the same WikiText-103 data pipeline and GPT-2 tokenizer stream as MOGT, with `ctx=32768`, `batch_size=1`, `grad_accum_steps=8`, and `200` optimizer steps.

Seed 42 was interrupted near step 190 before writing `last.pt` or the final JSON. The completed seed 42 report resumes from `baseline_checkpoints/mamba_scratch_budget_v1_ctx32768_seed42/best.pt`, which was saved at global step 50. The old checkpoint did not include scheduler state, so the resume path fast-forwarded the cosine scheduler by the saved global step. Newer `latest.pt` and `last.pt` checkpoints include scheduler state. Seeds 7 and 123 completed from scratch without resume.

## Main Comparison

| Model | Tokenization | Context | Config | Seeds | Steps | Validation batches | Mean loss | Loss std | Mean PPL | PPL std | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| MOGT `baseline_v1` Cayley | GPT-2 stream | 32768 | 12L, `d_model=768`, `rank=16`, `triton_hybrid` | 3 | 200 | 5 per seed | 6.4067 | 0.0878 | 607.45 | 54.41 | Mean checkpoint-only eval across seeds `42/7/123` |
| Scratch Mamba SSM | GPT-2 stream | 32768 | 24L, `d_model=768`, 129.12M params | 3 | 200 | 5 per seed | 9.6168 | 0.0949 | 15059.19 | 1423.11 | Best validation point per seed |

## Scratch Mamba Per-Seed Results

| Seed | Best loss | Best PPL | Final loss | Final PPL | Validation batches | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 9.5201 | 13631.30 | 9.5240 | 13684.71 | 5 | Resumed from step-50 best checkpoint after interruption |
| 7 | 9.7097 | 16477.48 | 9.7160 | 16581.26 | 5 | Completed from scratch |
| 123 | 9.6204 | 15068.78 | 9.6266 | 15162.54 | 5 | Completed from scratch |

## Scratch Mamba Validation Traces

| Seed | Step | Loss | PPL | Validation batches |
|---:|---:|---:|---:|---:|
| 42 | 50 | 9.5201 | 13631.30 | 5 |
| 42 | 100 | 9.5240 | 13684.30 | 5 |
| 42 | 150 | 9.5242 | 13686.96 | 5 |
| 42 | 200 | 9.5240 | 13684.71 | 5 |
| 7 | 50 | 9.7097 | 16477.48 | 5 |
| 7 | 100 | 9.7159 | 16579.86 | 5 |
| 7 | 150 | 9.7159 | 16578.54 | 5 |
| 7 | 200 | 9.7160 | 16581.26 | 5 |
| 123 | 50 | 9.6204 | 15068.78 | 5 |
| 123 | 100 | 9.6264 | 15159.31 | 5 |
| 123 | 150 | 9.6267 | 15163.63 | 5 |
| 123 | 200 | 9.6266 | 15162.54 | 5 |

Interpretation: under the same tokenizer, WikiText-103 stream, context length, and optimizer-step budget, a scratch Mamba-style baseline does not approach the current MOGT `baseline_v1` validation loss. This strengthens the short-budget training comparison, but it remains a narrow 200-step negative control rather than a complete language-modeling benchmark.
