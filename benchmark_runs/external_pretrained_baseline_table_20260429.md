# External Pretrained Baseline Anchors

These are pretrained reference anchors, not training-budget-matched baselines.

MOGT `baseline_v1` has only trained for 200 optimizer steps on WikiText-103 with the repo's GPT-2 tokenizer. GPT-2 uses the same GPT-2 token stream, but only at `ctx=1023` because of its positional limit. Mamba is evaluated with its own tokenizer on the raw WikiText-103 validation text, so its token-level PPL is not directly tokenization-matched to MOGT.

## Pretrained Anchors

| Model | Tokenization mode | Context | Validation batches | Loss | PPL | Notes |
|---|---|---:|---:|---:|---:|---|
| GPT-2 Small | GPT-2 stream | 1023 | 20 | 3.2287 | 25.25 | Tokenization-compatible with MOGT, but shorter context and pretrained |
| Mamba-130M HF | Native text | 8192 | 20 | 2.9984 | 20.05 | Native Mamba tokenizer; pretrained anchor |
| Mamba-130M HF | Native text | 16384 | 15 | 3.2028 | 24.60 | Native Mamba tokenizer; pretrained anchor |
| Mamba-130M HF | Native text | 32768 | 7 | 4.0787 | 59.07 | Native Mamba tokenizer; pretrained anchor |

## Current MOGT baseline_v1 For Context

Source: `benchmark_runs/baseline_v1_cayley_eval_table_20260429.md`

| Model | Tokenization mode | Context | Seeds | Validation batches per seed | Mean loss | Mean PPL | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| MOGT baseline_v1 Cayley | GPT-2 stream | 8192 | 3 | 20 | 6.4065 | 607.33 | 200 optimizer steps |
| MOGT baseline_v1 Cayley | GPT-2 stream | 16384 | 3 | 13 | 6.4186 | 614.73 | 200 optimizer steps |
| MOGT baseline_v1 Cayley | GPT-2 stream | 32768 | 3 | 5 | 6.4067 | 607.45 | 200 optimizer steps |

Interpretation: pretrained GPT-2 and Mamba are much stronger language models, as expected. This table should not be used as a fairness claim. Its value is to give scale and sanity anchors while the MOGT path is still validating operator correctness, training stability, and long-context feasibility.
