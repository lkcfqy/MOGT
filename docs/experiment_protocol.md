# Experiment Protocol

Last updated: 2026-05-04

## Purpose

Every result used in the paper must be reproducible, comparable, and honest
about failures. This protocol defines the minimum metadata and reporting format
for MOGT/GMAR experiments.

## Report Schema

Every JSON report should include these top-level fields:

Machine-readable schema: `docs/experiment_report_schema.json`.

```json
{
  "schema_version": "mogt-experiment-v1",
  "run_name": "",
  "task": "",
  "model": "",
  "variant": "",
  "status": "ok",
  "failure_reason": null,
  "command": "",
  "git_commit": "",
  "environment": {
    "device": "",
    "gpu_name": "",
    "torch_version": "",
    "cuda_available": true,
    "amp_dtype": "",
    "peak_memory_mb": null,
    "elapsed_seconds": null
  },
  "data": {
    "dataset": "",
    "tokenizer": "",
    "train_context": null,
    "eval_contexts": [],
    "train_examples": null,
    "eval_examples": null,
    "tokens_seen": null
  },
  "model_config": {
    "num_params": null,
    "d_model": null,
    "num_layers": null,
    "rank": null,
    "num_heads": null,
    "scan_impl": null,
    "connection_impl": null,
    "gate_config": null
  },
  "training": {
    "seed": null,
    "steps": null,
    "batch_size": null,
    "grad_accum_steps": null,
    "optimizer": "",
    "lr": null,
    "weight_decay": null,
    "scheduler": ""
  },
  "metrics": {
    "loss": null,
    "ppl": null,
    "accuracy_by_context": {},
    "loss_by_context": {}
  },
  "notes": ""
}
```

Allowed `status` values:

- `ok`
- `failed`
- `oom`
- `skipped`
- `partial`

## Main Table Rules

For every table in the paper:

- Include at least 3 seeds unless the table is explicitly labeled as a smoke or
  stress probe.
- Report mean and sample standard deviation.
- State whether error bars are standard deviation or standard error.
- Use the same number of eval examples for compared models.
- Include failed/OOM rows in system tables.
- Do not mix RoPE and NoPE baselines without explaining the positional
  extrapolation difference.

## Synthetic State-Tracking Rules

Required baselines:

- MOGT/GMAR proposed variant
- MOGT/GMAR without the proposed mechanism
- Transformer RoPE
- Transformer NoPE
- at least one SSM/recurrent baseline where practical

Required ablations:

- no gate
- value gate only
- independent gates
- coupled write-forget gate
- with and without prefix-conditioned addressing
- direct training and curriculum when curriculum is used

Required eval contexts:

- train context
- 2x train context
- 4x train context
- 8x train context
- larger stress context if memory allows

## Language Modeling Rules

Required baselines:

- scratch Transformer
- scratch Mamba or Mamba-style SSM
- MOGT/GMAR
- hybrid attention + MOGT when implemented

Rules:

- Match tokenizer and token stream for scratch comparisons.
- Report tokens seen, not only optimizer steps.
- Use at least 3 seeds before making quality claims.
- Pretrained HF models are anchors, not fair scratch baselines.

## Systems Rules

Required metrics:

- wall-clock per step
- peak memory
- operator profile
- context length
- batch size
- scan implementation
- loss implementation

Required comparisons:

- sequential reference
- block reference
- triton hybrid
- attention core / SDPA / FlashAttention where available
- Mamba/SSM baseline where available

## Evidence Labels

Use these labels consistently:

- `main_result`: supports a paper claim.
- `ablation`: isolates a mechanism.
- `baseline`: fairness control.
- `smoke`: checks that a path runs.
- `stress_probe`: pushes context/scale with weaker statistics.
- `negative`: important failed result.

## Promotion Rule

A result can move from `smoke` or `stress_probe` to `main_result` only when:

- it has 3+ seeds,
- the baseline protocol is matched,
- the command is recorded,
- and the JSON artifact follows this protocol.
