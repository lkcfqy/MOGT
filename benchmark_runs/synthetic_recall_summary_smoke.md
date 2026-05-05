# Synthetic Recall Smoke Summary

Date: 2026-05-03

This is a pipeline smoke test for `benchmark_synthetic_recall.py`, not a model
quality result. Both models were trained for only 2 steps, so zero recall
accuracy is expected.

## Smoke Runs

| Model | Params | Train ctx | Eval ctx | Pairs | Steps | Eval batches | Peak MB | Train elapsed | Status |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| MOGT | 1,049,472 | 512 | 512, 1024 | 4 | 2 | 2 | 59.94 | 1.91s | ok |
| Transformer | 1,049,216 | 512 | 512, 1024 | 4 | 2 | 2 | 51.46 | 0.55s | ok |

## Eval Results

| Model | Context | Depth | Loss | Accuracy |
|---|---:|---:|---:|---:|
| MOGT | 512 | 0.5 | 8.5654 | 0.000 |
| MOGT | 1024 | 0.5 | 8.4892 | 0.000 |
| Transformer | 512 | 0.5 | 8.3867 | 0.000 |
| Transformer | 1024 | 0.5 | 8.3818 | 0.000 |

Artifacts:

- `benchmark_runs/synthetic_recall_mogt_smoke.json`
- `benchmark_runs/synthetic_recall_transformer_smoke.json`

## Suggested Next Sweep

Small 3-seed learning sanity:

```bash
python3 benchmark_synthetic_recall.py --model-type mogt --train-context 1024 --eval-contexts 1024 2048 4096 --eval-depths 0.1 0.5 0.9 --d-model 256 --num-layers 4 --rank 16 --steps 1000 --batch-size 8 --eval-batches 32 --num-pairs 8 --device cuda --scan-impl triton_hybrid --output benchmark_runs/synthetic_recall_mogt_ctx1024_seed42.json
python3 benchmark_synthetic_recall.py --model-type transformer --train-context 1024 --eval-contexts 1024 2048 4096 --eval-depths 0.1 0.5 0.9 --d-model 256 --num-layers 4 --num-heads 4 --steps 1000 --batch-size 8 --eval-batches 32 --num-pairs 8 --device cuda --output benchmark_runs/synthetic_recall_transformer_ctx1024_seed42.json
```

Long-context paper probe after the small sweep learns:

```bash
python3 benchmark_synthetic_recall.py --model-type mogt --train-context 8192 --eval-contexts 8192 32768 65536 --eval-depths 0.1 0.5 0.9 --d-model 512 --num-layers 8 --rank 16 --steps 3000 --batch-size 2 --eval-batches 32 --num-pairs 16 --device cuda --scan-impl triton_hybrid --gradient-checkpointing --output benchmark_runs/synthetic_recall_mogt_ctx8192_seed42.json
python3 benchmark_synthetic_recall.py --model-type transformer --train-context 8192 --eval-contexts 8192 32768 65536 --eval-depths 0.1 0.5 0.9 --d-model 512 --num-layers 8 --num-heads 8 --steps 3000 --batch-size 2 --eval-batches 32 --num-pairs 16 --device cuda --gradient-checkpointing --output benchmark_runs/synthetic_recall_transformer_ctx8192_seed42.json
```

Interpretation rule: if the small sweep does not learn the in-distribution
1024-context task, do not scale it yet. First tune learning rate, task difficulty,
or architecture size.
