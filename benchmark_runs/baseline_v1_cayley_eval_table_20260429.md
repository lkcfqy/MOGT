# baseline_v1 Cayley Checkpoint-Only Evaluation

Source: `benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json`

Configuration: `triton_hybrid`, `cayley`, `d_model=768`, `layers=12`, `rank=16`, `batch_size=1`, seeds `42/7/123`.

| Context | Validation batches per seed | Mean loss | Loss std | Mean PPL | PPL std |
|---:|---:|---:|---:|---:|---:|
| 8192 | 20 | 6.4065 | 0.0867 | 607.33 | 53.72 |
| 16384 | 13 | 6.4186 | 0.0866 | 614.73 | 54.31 |
| 32768 | 5 | 6.4067 | 0.0878 | 607.45 | 54.41 |

Per-seed results:

| Seed | Context | Loss | PPL | Validation batches |
|---:|---:|---:|---:|---:|
| 42 | 8192 | 6.3408 | 567.23 | 20 |
| 42 | 16384 | 6.3540 | 574.78 | 13 |
| 42 | 32768 | 6.3407 | 567.22 | 5 |
| 7 | 8192 | 6.3740 | 586.38 | 20 |
| 7 | 16384 | 6.3849 | 592.83 | 13 |
| 7 | 32768 | 6.3729 | 585.77 | 5 |
| 123 | 8192 | 6.5048 | 668.37 | 20 |
| 123 | 16384 | 6.5170 | 676.57 | 13 |
| 123 | 32768 | 6.5063 | 669.36 | 5 |

Interpretation: the wider checkpoint-only evaluation does not invalidate the earlier `ctx=32768` five-batch signal. It supports short-run training stability for the current `baseline_v1` candidate, but it is still not a complete language-modeling benchmark because the model has only trained for 200 optimizer steps.
