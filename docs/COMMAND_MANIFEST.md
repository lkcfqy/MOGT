# Command Manifest

Last updated: 2026-05-05

This file records commands that generate or verify the evidence used by the
current paper direction. Add new commands here before launching long runs.

## Sanity Checks

```bash
python3 -m compileall -q .
python3 sanity_affine_scan.py
python3 sanity_triton_gradients.py
python3 sanity_triton_training.py
python3 validate_experiment_reports.py
```

## Historical MOGT 32k Baseline

The legacy all-in-one LM runner was removed from the clean GitHub handoff.
The useful evidence is preserved as reports:

- `benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json`
- `benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json`
- `benchmark_runs/baseline_v1_cayley_eval_table_20260429.md`

Checkpoint-only eval:

```bash
python3 evaluate_checkpoints.py \
  --checkpoint \
    mogt_checkpoints/baseline_v1_cayley_ctx32768_seed42 \
    mogt_checkpoints/baseline_v1_cayley_ctx32768_seed7 \
    mogt_checkpoints/baseline_v1_cayley_ctx32768_seed123 \
  --context-lengths 8192 16384 32768 \
  --max-batches 20 \
  --output benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json
```

## Scratch Baselines

Mamba-style baseline:

```bash
python3 train_budget_baseline.py \
  --run-name mamba_scratch_budget_v1_ctx32768_seed42 \
  --context-length 32768 \
  --d-model 768 \
  --num-layers 24 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-steps 200 \
  --eval-interval 50 \
  --eval-max-batches 10 \
  --seed 42 \
  --report-output benchmark_runs/mamba_scratch_budget_v1_ctx32768_seed42.json \
  --checkpoint-dir baseline_checkpoints/mamba_scratch_budget_v1_ctx32768_seed42
```

Transformer baseline:

```bash
python3 train_budget_transformer.py \
  --run-name transformer_scratch_budget_v1_ctx32768_seed42 \
  --context-length 32768 \
  --d-model 768 \
  --num-layers 12 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-steps 200 \
  --eval-interval 50 \
  --eval-max-batches 10 \
  --seed 42 \
  --report-output benchmark_runs/transformer_scratch_budget_v1_ctx32768_seed42.json \
  --checkpoint-dir baseline_checkpoints/transformer_scratch_budget_v1_ctx32768_seed42
```

Hybrid MOGT/Transformer pilot:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_alt_ctx8192_d128_l2_steps10_seed42 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --rank 16 \
  --hybrid-pattern alternating \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 10 \
  --eval-interval 5 \
  --eval-max-batches 2 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --report-output benchmark_runs/hybrid_alt_ctx8192_d128_l2_steps10_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_alt_ctx8192_d128_l2_steps10_seed42
```

Matched tiny Transformer pilot:

```bash
python3 train_budget_transformer.py \
  --run-name transformer_ctx8192_d128_l2_steps10_seed42 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 10 \
  --eval-interval 5 \
  --eval-max-batches 2 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --report-output benchmark_runs/transformer_ctx8192_d128_l2_steps10_seed42.json \
  --checkpoint-dir baseline_checkpoints/transformer_ctx8192_d128_l2_steps10_seed42
```

Hybrid layer-ratio pilot:

```bash
python3 run_hybrid_lm_sweep.py \
  --run-prefix hybrid_ratio_sweep_pilot \
  --fractions 0 0.25 0.5 0.75 1 \
  --seeds 42 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 4 \
  --rank 16 \
  --max-steps 5 \
  --eval-interval 5 \
  --eval-max-batches 1 \
  --num-workers 0
python3 summarize_hybrid_lm_sweep.py
```

Initialization-control variant:

```bash
python3 run_hybrid_lm_sweep.py \
  --run-prefix hybrid_ratio_sweep_zeroattn_pilot \
  --fractions 0 0.25 0.5 0.75 1 \
  --seeds 42 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 4 \
  --rank 16 \
  --max-steps 5 \
  --eval-interval 5 \
  --eval-max-batches 1 \
  --num-workers 0 \
  --zero-init-attention-out
python3 summarize_hybrid_lm_sweep.py
```

Longer zero-init pilot:

```bash
python3 run_hybrid_lm_sweep.py \
  --run-prefix hybrid_ratio_sweep_zeroattn_steps50 \
  --fractions 0 0.25 1 \
  --seeds 42 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 4 \
  --rank 16 \
  --max-steps 50 \
  --eval-interval 25 \
  --eval-max-batches 2 \
  --num-workers 0 \
  --zero-init-attention-out
python3 summarize_hybrid_lm_sweep.py
```

Three-seed 50-step zero-init pilot:

```bash
python3 run_hybrid_lm_sweep.py \
  --run-prefix hybrid_ratio_sweep_zeroattn_steps50_3seed \
  --fractions 0 0.25 0.5 \
  --seeds 7 42 123 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 4 \
  --rank 16 \
  --max-steps 50 \
  --eval-interval 25 \
  --eval-max-batches 2 \
  --num-workers 0 \
  --zero-init-attention-out
python3 summarize_hybrid_lm_sweep.py
```

Three-seed 200-step zero-init follow-up:

```bash
python3 run_hybrid_lm_sweep.py \
  --run-prefix hybrid_ratio_sweep_zeroattn_steps200 \
  --fractions 0 0.25 \
  --seeds 7 42 123 \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 4 \
  --rank 16 \
  --max-steps 200 \
  --eval-interval 100 \
  --eval-max-batches 4 \
  --num-workers 0 \
  --zero-init-attention-out
python3 summarize_hybrid_lm_sweep.py
```

Single-MOGT-layer position follow-up at the same budget:

```bash
set -e
for layer in 0 2 3; do
  for seed in 7 42 123; do
    if [ "$layer" = "0" ] && [ "$seed" != "42" ]; then
      continue
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_layeridx_steps200_l${layer}_ctx8192_d128_l4_seed${seed} \
      --context-length 8192 \
      --d-model 128 \
      --num-layers 4 \
      --num-heads 4 \
      --rank 16 \
      --seed ${seed} \
      --mogt-layer-indices ${layer} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 200 \
      --eval-interval 100 \
      --eval-max-batches 4 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_layeridx_steps200_l${layer}_ctx8192_d128_l4_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_layeridx_steps200_l${layer}_ctx8192_d128_l4_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py
python3 summarize_paper_results.py
```

Next late-layer scale-up target:

```bash
set -e
for variant in attn l2 l3 l12; do
  for seed in 7 42 123; do
    if [ "$variant" = "attn" ]; then
      extra_args="--mogt-layer-fraction 0"
    elif [ "$variant" = "l2" ]; then
      extra_args="--mogt-layer-indices 2"
    elif [ "$variant" = "l12" ]; then
      extra_args="--mogt-layer-indices 1 2"
    else
      extra_args="--mogt-layer-indices 3"
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_late_l3_steps500_${variant}_ctx8192_d128_l4_seed${seed} \
      --context-length 8192 \
      --d-model 128 \
      --num-layers 4 \
      --num-heads 4 \
      --rank 16 \
      --seed ${seed} \
      ${extra_args} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 500 \
      --eval-interval 250 \
      --eval-max-batches 8 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_late_l3_steps500_${variant}_ctx8192_d128_l4_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_late_l3_steps500_${variant}_ctx8192_d128_l4_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py \
  --steps 500 \
  --output-md benchmark_runs/hybrid_layer_position_steps500_summary_20260504.md
```

1000-step continuation of the same late-layer test:

```bash
set -e
for variant in attn l3; do
  for seed in 7 42 123; do
    if [ "$variant" = "attn" ]; then
      extra_args="--mogt-layer-fraction 0"
    else
      extra_args="--mogt-layer-indices 3"
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_late_l3_steps1000_${variant}_ctx8192_d128_l4_seed${seed} \
      --context-length 8192 \
      --d-model 128 \
      --num-layers 4 \
      --num-heads 4 \
      --rank 16 \
      --seed ${seed} \
      ${extra_args} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 1000 \
      --eval-interval 500 \
      --eval-max-batches 8 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_late_l3_steps1000_${variant}_ctx8192_d128_l4_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_late_l3_steps1000_${variant}_ctx8192_d128_l4_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py \
  --steps 1000 \
  --output-md benchmark_runs/hybrid_layer_position_steps1000_summary_20260504.md
python3 summarize_paper_results.py
```

First width scale probe:

```bash
set -e
for variant in attn l3; do
  for seed in 7 42 123; do
    if [ "$variant" = "attn" ]; then
      extra_args="--mogt-layer-fraction 0"
    else
      extra_args="--mogt-layer-indices 3"
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_scale_d192_l4_steps500_${variant}_ctx8192_seed${seed} \
      --context-length 8192 \
      --d-model 192 \
      --num-layers 4 \
      --num-heads 6 \
      --rank 16 \
      --seed ${seed} \
      ${extra_args} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 500 \
      --eval-interval 250 \
      --eval-max-batches 8 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_scale_d192_l4_steps500_${variant}_ctx8192_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps500_${variant}_ctx8192_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py \
  --d-model 192 \
  --steps 500 \
  --output-md benchmark_runs/hybrid_scale_d192_l4_steps500_summary_20260504.md
python3 summarize_paper_results.py
```

d192 1000-step boundary check:

```bash
set -e
for variant in attn l3; do
  for seed in 7 42 123; do
    if [ "$variant" = "attn" ]; then
      extra_args="--mogt-layer-fraction 0"
    else
      extra_args="--mogt-layer-indices 3"
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_scale_d192_l4_steps1000_${variant}_ctx8192_seed${seed} \
      --context-length 8192 \
      --d-model 192 \
      --num-layers 4 \
      --num-heads 6 \
      --rank 16 \
      --seed ${seed} \
      ${extra_args} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 1000 \
      --eval-interval 500 \
      --eval-max-batches 8 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_${variant}_ctx8192_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_${variant}_ctx8192_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py \
  --d-model 192 \
  --steps 1000 \
  --output-md benchmark_runs/hybrid_scale_d192_l4_steps1000_summary_20260504.md
python3 summarize_paper_results.py
```

d192 1000-step lr=5e-4 fairness check:

```bash
set -e
for variant in attn l2; do
  for seed in 7 42 123; do
    if [ "$variant" = "attn" ]; then
      extra_args="--mogt-layer-fraction 0"
    else
      extra_args="--mogt-layer-indices 2"
    fi
    python3 train_budget_hybrid.py \
      --run-name hybrid_scale_d192_l4_steps1000_${variant}_lr5e4_ctx8192_seed${seed} \
      --context-length 8192 \
      --d-model 192 \
      --num-layers 4 \
      --num-heads 6 \
      --rank 16 \
      --seed ${seed} \
      ${extra_args} \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --max-steps 1000 \
      --eval-interval 500 \
      --eval-max-batches 8 \
      --num-workers 0 \
      --latest-checkpoint-interval 0 \
      --zero-init-attention-out \
      --lr 0.0005 \
      --no-save-best \
      --no-save-last \
      --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_${variant}_lr5e4_ctx8192_seed${seed}.json \
      --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_${variant}_lr5e4_ctx8192_seed${seed}
  done
done
python3 summarize_hybrid_layer_positions.py \
  --d-model 192 \
  --steps 1000 \
  --lr 0.0005 \
  --output-md benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_summary_20260504.md
python3 summarize_paper_results.py
```

d192 1000-step residual-scale diagnostic:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 192 \
  --num-layers 4 \
  --num-heads 6 \
  --rank 16 \
  --seed 42 \
  --mogt-layer-indices 2 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-ffn-residual-scale 1.0 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_ctx8192_seed42
python3 summarize_hybrid_layer_positions.py \
  --d-model 192 \
  --steps 1000 \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --output-md benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_summary_20260505.md
python3 summarize_paper_results.py
```

Follow-up seeds `7` and `123` were run with the same command shape, replacing
`--seed`, `--run-name`, `--report-output`, and `--checkpoint-dir` with the
matching seed-specific suffixes. The resulting three-seed scale-0.5 summary is
`benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_summary_20260505.md`:
attention-only is `5.8753 +/- 0.0108`, layer2 MOGT scale `0.5` is
`5.8775 +/- 0.0293`, and the scaled hybrid beats attention on seeds `7` and
`123` but remains `0.0022` worse in aggregate.

Seed-42 fixed-scale sweep around the same run:

```bash
# Repeat the command above with:
#   --mogt-residual-scale 0.25
#   --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p25_ctx8192_seed42
#   --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p25_ctx8192_seed42.json
#   --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p25_ctx8192_seed42
#
# and:
#   --mogt-residual-scale 0.75
#   --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p75_ctx8192_seed42
#   --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p75_ctx8192_seed42.json
#   --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p75_ctx8192_seed42
```

Learned residual-gate smoke:

```bash
python3 train_budget_hybrid.py \
  --smoke \
  --run-name hybrid_residual_gate_smoke \
  --context-length 8192 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --rank 16 \
  --mogt-layer-indices 1 \
  --mogt-residual-gate \
  --mogt-residual-gate-init 0.5 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1 \
  --eval-interval 1 \
  --eval-max-batches 1 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_residual_gate_smoke.json \
  --checkpoint-dir baseline_checkpoints/hybrid_residual_gate_smoke
```

Next learned residual-gate budget diagnostic:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtgate0p5_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 192 \
  --num-layers 4 \
  --num-heads 6 \
  --rank 16 \
  --seed 42 \
  --mogt-layer-indices 2 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --mogt-residual-gate \
  --mogt-residual-gate-init 0.5 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtgate0p5_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtgate0p5_ctx8192_seed42
```

Result note: this naive learned gate is negative in seed42. It reaches val loss
`5.9752`, worse than fixed residual scale `0.5` at `5.9110`.

Residual-scale schedule diagnostic:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtsched0p25to0p5s250_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 192 \
  --num-layers 4 \
  --num-heads 6 \
  --rank 16 \
  --seed 42 \
  --mogt-layer-indices 2 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-residual-scale-start 0.25 \
  --mogt-residual-scale-warmup-steps 250 \
  --mogt-ffn-residual-scale 1.0 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtsched0p25to0p5s250_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtsched0p25to0p5s250_ctx8192_seed42
```

Result note: this simple schedule is negative in seed42. It reaches val loss
`5.9497`, worse than fixed residual scale `0.5` at `5.9110`.

MOGT learning-rate multiplier diagnostic:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 192 \
  --num-layers 4 \
  --num-heads 6 \
  --rank 16 \
  --seed 42 \
  --mogt-layer-indices 2 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-lr-mult 0.5 \
  --mogt-ffn-residual-scale 1.0 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d192_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42
```

Result note: this optimizer-partition diagnostic is positive in seed42. It
reaches val loss `5.8601`, better than same-seed attention `5.8877` and fixed
scale `0.5` with default MOGT LR at `5.9110`.

Follow-up seeds `7` and `123` were run with the same command shape, replacing
`--seed`, `--run-name`, `--report-output`, and `--checkpoint-dir` with the
matching seed-specific suffixes. Summary command:

```bash
python3 summarize_hybrid_layer_positions.py \
  --d-model 192 \
  --steps 1000 \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-lr-mult 0.5 \
  --output-md benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_20260505.md
python3 summarize_paper_results.py
```

Result note: the three-seed aggregate is positive. Attention-only is
`5.8753 +/- 0.0108`; layer2 MOGT with residual scale `0.5` and
`mogt_lr_mult=0.5` is `5.8511 +/- 0.0117`, winning on all paired seeds.

d256 seed42 width migration check:

```bash
python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d256_l4_steps1000_attn_lr5e4_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --rank 16 \
  --seed 42 \
  --hybrid-pattern all_transformer \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d256_l4_steps1000_attn_lr5e4_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d256_l4_steps1000_attn_lr5e4_ctx8192_seed42

python3 train_budget_hybrid.py \
  --run-name hybrid_scale_d256_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42 \
  --context-length 8192 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --rank 16 \
  --seed 42 \
  --mogt-layer-indices 2 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1000 \
  --eval-interval 500 \
  --eval-max-batches 8 \
  --num-workers 0 \
  --latest-checkpoint-interval 0 \
  --zero-init-attention-out \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-lr-mult 0.5 \
  --mogt-ffn-residual-scale 1.0 \
  --no-save-best \
  --no-save-last \
  --report-output benchmark_runs/hybrid_scale_d256_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42.json \
  --checkpoint-dir baseline_checkpoints/hybrid_scale_d256_l4_steps1000_l2_lr5e4_mogtscale0p5_mogtlr0p5_ctx8192_seed42

python3 summarize_hybrid_layer_positions.py \
  --d-model 256 \
  --steps 1000 \
  --lr 0.0005 \
  --mogt-residual-scale 0.5 \
  --mogt-lr-mult 0.5 \
  --output-md benchmark_runs/hybrid_scale_d256_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_seed42_20260505.md
python3 summarize_paper_results.py
```

Result note: this width migration check is neutral. At seed42, attention-only
reaches `5.7550`; layer2 MOGT with residual scale `0.5` and
`mogt_lr_mult=0.5` reaches `5.7558`.

## Synthetic Last-Value Tracking

Use existing summary artifacts as the current baseline of record:

```bash
python3 summarize_synthetic_last_value.py
```

Matched NoPE Transformer, train context 128, 2048 eval examples per context:

```bash
python3 benchmark_synthetic_last_value.py \
  --model-type transformer \
  --rope-theta 0.0 \
  --seed 7 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --train-context 128 \
  --eval-contexts 128 256 512 1024 \
  --batch-size 64 \
  --steps 2000 \
  --eval-batches 32 \
  --log-every 500 \
  --output benchmark_runs/synthetic_last_value_transformer_nope_matchedeval_ctx128_seed7_steps2000.json
```

Repeat with seeds `42` and `123`.

Standard train-512 single-slot main table:

```bash
python3 benchmark_synthetic_last_value.py \
  --model-type mogt \
  --seed 7 \
  --d-model 128 \
  --num-layers 2 \
  --rank 16 \
  --scan-impl triton_hybrid \
  --connection-impl identity \
  --connection-damping 1.0 \
  --value-gate \
  --value-gate-bias -2.0 \
  --couple-forget-to-value-gate \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 8192 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 2000 \
  --eval-batches 16 \
  --dense-loss \
  --log-every 400 \
  --output benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_stdreport_biasm2_ctx512_seed7_steps2000.json
```

Repeat with seeds `42` and `123`.

```bash
python3 benchmark_synthetic_last_value.py \
  --model-type transformer \
  --rope-theta 0.0 \
  --seed 7 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 8192 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 2000 \
  --eval-batches 16 \
  --dense-loss \
  --log-every 400 \
  --output benchmark_runs/synthetic_last_value_transformer_nope_dense_stdreport_ctx512_seed7_steps2000.json
```

Repeat with seeds `42` and `123`.

## Synthetic Multi-Slot Tracking

Use existing summary artifacts as the current baseline of record:

```bash
python3 summarize_synthetic_multislot.py
python3 summarize_standard_multislot.py
python3 summarize_paper_results.py
```

Standard 4-slot final-query-only curriculum table:

```bash
python3 benchmark_synthetic_multislot.py \
  --model-type mogt \
  --seed 7 \
  --vocab-size 160 \
  --d-model 128 \
  --num-layers 2 \
  --rank 16 \
  --scan-impl triton_hybrid \
  --connection-impl identity \
  --connection-damping 1.0 \
  --readout-init-std 0.02 \
  --value-gate \
  --value-gate-width rank \
  --value-gate-input current_prev_prefix \
  --value-gate-bias -2.0 \
  --couple-forget-to-value-gate \
  --prefix-condition-position 1 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 \
  --num-slots 4 \
  --min-train-slots 2 \
  --slot-curriculum-steps 1500 \
  --num-values 16 \
  --min-updates 4 \
  --max-updates 12 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 3000 \
  --eval-batches 16 \
  --log-every 500 \
  --output benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json
```

Repeat with seeds `42` and `123`.

```bash
python3 benchmark_synthetic_multislot.py \
  --model-type transformer \
  --rope-theta 0.0 \
  --seed 7 \
  --vocab-size 160 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 \
  --num-slots 4 \
  --min-train-slots 2 \
  --slot-curriculum-steps 1500 \
  --num-values 16 \
  --min-updates 4 \
  --max-updates 12 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 3000 \
  --eval-batches 16 \
  --log-every 500 \
  --output benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json
```

Repeat with seeds `42` and `123`.

Parameter-matched HF-Mamba d192 baseline:

```bash
python3 benchmark_synthetic_multislot.py \
  --model-type mamba \
  --seed 7 \
  --vocab-size 160 \
  --d-model 192 \
  --num-layers 2 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 \
  --num-slots 4 \
  --min-train-slots 2 \
  --slot-curriculum-steps 1500 \
  --num-values 16 \
  --min-updates 4 \
  --max-updates 12 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 3000 \
  --eval-batches 16 \
  --log-every 500 \
  --output benchmark_runs/synthetic_multislot4_mamba_d192_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json
```

Repeat with seeds `42` and `123`.

Direct/no-curriculum ablation:

```bash
python3 benchmark_synthetic_multislot.py \
  --model-type mogt \
  --seed 7 \
  --vocab-size 160 \
  --d-model 128 \
  --num-layers 2 \
  --rank 16 \
  --scan-impl triton_hybrid \
  --connection-impl identity \
  --connection-damping 1.0 \
  --readout-init-std 0.02 \
  --value-gate \
  --value-gate-width rank \
  --value-gate-input current_prev_prefix \
  --value-gate-bias -2.0 \
  --couple-forget-to-value-gate \
  --prefix-condition-position 1 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 \
  --num-slots 4 \
  --num-values 16 \
  --min-updates 4 \
  --max-updates 12 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 3000 \
  --eval-batches 16 \
  --log-every 500 \
  --output benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_finalonly_stdreport_ctx512_seed7_steps3000.json
```

Repeat with seeds `42` and `123`.

```bash
python3 benchmark_synthetic_multislot.py \
  --model-type transformer \
  --rope-theta 0.0 \
  --seed 7 \
  --vocab-size 160 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --train-context 512 \
  --eval-contexts 512 1024 2048 4096 \
  --num-slots 4 \
  --num-values 16 \
  --min-updates 4 \
  --max-updates 12 \
  --batch-size 16 \
  --eval-batch-size 4 \
  --steps 3000 \
  --eval-batches 16 \
  --log-every 500 \
  --output benchmark_runs/synthetic_multislot4_transformer_nope_finalonly_stdreport_ctx512_seed7_steps3000.json
```

Repeat with seeds `42` and `123`.

Next required rerun:

- Keep NoPE Transformer first-class in every table.
- If direct/no-curriculum becomes a main table, add stronger recurrent
  baselines for that setting too.

## Systems

Core operator timing is preserved as:

- `benchmark_runs/throughput_core_operator_d768_len8192_16384_32768_20260504.json`
- `benchmark_runs/throughput_core_operator_summary_20260504.md`

The clean handoff keeps the backbone-level timing script as the current systems
smoke path.

Backbone forward probe:

```bash
python3 benchmark_backbone_throughput.py \
  --d-model 768 \
  --num-layers 2 \
  --batch-size 1 \
  --lengths 8192 16384 32768 \
  --warmup 1 \
  --iters 3 \
  --value-gate \
  --couple-forget-to-value-gate \
  --output-json benchmark_runs/backbone_throughput_identity_coupled_d768_l2_len8192_16384_32768_20260504.json
python3 summarize_backbone_throughput.py
```

Training profile:

```bash
MOGT_RUN_PRESET=baseline_v1_smoke python3 profile_train_step.py
```

## Before Any Submission Draft

Run:

```bash
python3 -m compileall -q .
python3 sanity_affine_scan.py
python3 sanity_triton_gradients.py
python3 sanity_triton_training.py
python3 validate_experiment_reports.py
```

Then regenerate:

```bash
python3 summarize_budget_baselines.py
python3 summarize_synthetic_last_value.py
python3 summarize_synthetic_multislot.py
python3 summarize_standard_multislot.py
python3 summarize_paper_results.py
python3 summarize_throughput_results.py
python3 summarize_backbone_throughput.py
python3 summarize_hybrid_lm_sweep.py
python3 summarize_standard_reports.py
```
