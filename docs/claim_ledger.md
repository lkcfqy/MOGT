# MOGT Top-Tier Paper Claim Ledger

This ledger keeps the paper story honest. It separates claims that are already
supported by repository evidence from claims that still need experiments before
they can survive a serious systems/ML review.

## Current Status

MOGT is a working research prototype for a matrix-valued affine recurrence:

```text
H_t = U_t H_{t-1} + V_t
```

The repo now has correctness checks, multiple scan implementations, a WikiText
training path, a scratch Mamba baseline, a scratch Transformer baseline, and
early long-context memory probes. It also has an initial hybrid MOGT/Transformer
language-modeling prototype. The strongest current result is not that MOGT
beats Transformers on standard language-modeling quality. The scratch
Transformer baseline currently beats MOGT on the 32k, 200-step WikiText protocol.

## Claims We Can Make Now

1. MOGT implements a real non-attention sequence operator.
   - Evidence: `affine_scan.py`, `model_mogt.py`, `sanity_affine_scan.py`,
     `sanity_triton_gradients.py`, `sanity_triton_training.py`.

2. The affine recurrence has a clear associative scan structure.
   - Evidence: sequential, doubling, block-reference, and hybrid Triton scan
     paths are present in the repo.

3. The repo has moved from self-comparison to budget-matched baselines.
   - Evidence: `model_baseline_transformer.py`,
     `train_budget_transformer.py`, `model_baseline_hf_mamba.py`,
     `train_budget_baseline.py`, and
     `benchmark_runs/budget_matched_baseline_summary.md`.

4. The memory-efficient LM loss fixed a real long-context training blocker.
   - Evidence: `chunked_lm_loss.py` and
     `benchmark_runs/long_context_probe_summary_20260503.md`.

5. A gated affine-transport variant shows a real length-extrapolation signal on
   a controlled recurrent memory task.
   - Evidence: `benchmark_synthetic_last_value.py` and
     `benchmark_runs/synthetic_last_value_summary_20260503.md`.
   - Train context 128, eval context 1024, 3 seeds:
     gated MOGT accuracy is 69.71% +/- 18.69%, matched RoPE Transformer
     accuracy is 10.51% +/- 1.24%.
   - Matched-eval NoPE Transformer is a much stronger baseline than RoPE.
     With the same 2048 eval examples per context as the main table, NoPE
     Transformer reaches 99.07% +/- 0.32%, 98.10% +/- 0.85%,
     91.47% +/- 5.80%, and 68.64% +/- 15.51% at
     128/256/512/1024. This is close to gated MOGT at 1024 and means
     last-value train-context-128 should be framed as "RoPE extrapolation
     failure plus NoPE competitive baseline," not as a decisive attention win.
   - Scaling stress test caveat: direct train-context-512 currently fails for
     gated MOGT, while a NoPE Transformer learns the train-512 task and reaches
     90.62% at 2048 and 59.38% at 4096 in the seed-42 probe.
   - New train-512 diagnostic: switching MOGT to identity transport, adding a
     value/write gate, and using dense state supervision makes it learn quickly
     (all 3 seeds reach 100% at train context 512), but independently gated
     identity MOGT still trails NoPE Transformer at far extrapolation under
     the same dense supervision: at 8192, identity dual-gate MOGT is
     47.40% +/- 8.88%, while the current standard-schema NoPE rerun is
     78.65% +/- 12.63% and an earlier legacy NoPE run was
     89.06% +/- 7.16%.
   - New mechanism result: coupling the value/write gate to transport forgetting
     fixes the single-slot far-extrapolation failure under dense supervision.
     The standard-schema rerun uses `value_gate_bias=-2.0`; across seeds
     7/42/123, identity coupled write-forget MOGT reaches 100.00% +/- 0.00%
     at every evaluated context from 512 through 8192 after training only at
     context 512. Under the same current standard-schema rerun, NoPE
     Transformer reaches 78.65% +/- 12.63% at 8192.
   - Seed-42 long probe: the same coupled write-forget setup remains at 100%
     from context 512 through 65,536 with 16 eval examples per context. This is
     a stress probe, not yet a multi-seed statistical claim.
   - Gate diagnostic: in the seed-42 coupled model, block 0 opens the value gate
     to 88.83% on value tokens and approximately 0.0001% on SET/QUERY/filler
     tokens, directly matching the intended write-on-value, keep-otherwise
     mechanism. Block 1 is less selective, so the paper should describe this as
     a first-layer mechanism diagnostic rather than a claim about every layer.
   - Seed-42 gate ablation: identity value-gate-only MOGT learns the train
     context but drops to 48.44% at 4096 and 25.00% at 8192, worse than the
     dual-gate version. This supports the need for an explicit transport/forget
     gate. A residual transport-gate mode produced NaNs in the seed-42 probe;
     forget-ReLU was stable but worse at far extrapolation.
   - Important caveat: the original ungated MOGT stays near chance on the same
     overwrite-style task, so the paper should present this as a discovered
     architectural requirement rather than as a win for the original operator.

6. Prefix-conditioned slot-addressed gates make tracked multi-slot routing
   learnable in controlled dense-supervision probes.
   - Evidence: `benchmark_synthetic_multislot.py` and
     `benchmark_runs/synthetic_multislot_summary_20260503.md`.
   - Adding `current_prev_prefix` gate input (current token, previous token,
     and tracked-slot prefix) turns 2-slot routing into a strong positive
     result. Across seeds 7/42/123, MOGT reaches 100.00% +/- 0.00% at 512 and
     1024, 98.44% +/- 1.56% at 2048, and 86.46% +/- 3.25% at 4096.
   - The matched NoPE Transformer 3-seed baseline is 45.31% +/- 4.69%,
     41.15% +/- 3.93%, 34.90% +/- 3.25%, and 44.79% +/- 9.42%.
   - 4-slot 3-seed result: slot-addressed MOGT reaches
     100.00% +/- 0.00%, 100.00% +/- 0.00%, 100.00% +/- 0.00%, and
     98.96% +/- 0.90% at 512/1024/2048/4096. The matched NoPE Transformer
     reaches 23.96% +/- 3.61%, 25.00% +/- 4.13%, 21.35% +/- 4.77%, and
     24.48% +/- 7.05%.
   - 2-slot final-query-only 3-seed result: slot-addressed MOGT reaches
     100.00% +/- 0.00%, 100.00% +/- 0.00%, 100.00% +/- 0.00%, and
     96.35% +/- 3.25% at 512/1024/2048/4096. The matched NoPE Transformer
     reaches 42.19% +/- 2.71%, 40.62% +/- 7.16%, 47.92% +/- 3.93%, and
     42.71% +/- 21.67%.
   - Direct 4-slot final-only without curriculum is weak in the current
     standard-schema 3-seed rerun: MOGT reaches 44.79% +/- 17.21% at 4096 and
     NoPE reaches 25.00% +/- 5.63%.
   - 4-slot final-query-only with a 2-to-4-slot curriculum is now a 3-seed
     positive result. Slot-addressed MOGT reaches 100.00% +/- 0.00%,
     100.00% +/- 0.00%, 99.48% +/- 0.90%, and 94.27% +/- 2.39% at
     512/1024/2048/4096 in the current standard-schema rerun. The matched
     standard-schema NoPE Transformer reaches 31.25% +/- 5.41%,
     29.69% +/- 5.63%, 23.44% +/- 5.63%, and 21.35% +/- 6.31%.
     Parameter-matched HF-Mamba d192 reaches 17.71% +/- 10.97%,
     20.83% +/- 9.92%, 15.62% +/- 5.63%, and 19.79% +/- 10.97%.
   - Write-only supervision is highly seed-sensitive; dense-1500-to-final
     seed42 reaches 96.88% at 4096 for MOGT and 26.56% for NoPE.
   - 6-slot final-query-only with a 2-to-6-slot curriculum is also a 3-seed
     positive result. Slot-addressed MOGT reaches 100.00% +/- 0.00%,
     100.00% +/- 0.00%, 98.96% +/- 0.90%, and 96.88% +/- 2.71% at
     512/1024/2048/4096. The matched NoPE Transformer reaches
     20.83% +/- 1.80%, 22.92% +/- 5.02%, 21.35% +/- 5.49%, and
     21.88% +/- 8.27%.
   - 8-slot final-query-only with a 2-to-8-slot curriculum is positive but
     more variable. Slot-addressed MOGT reaches 100.00% +/- 0.00%,
     100.00% +/- 0.00%, 97.40% +/- 0.90%, and 85.42% +/- 11.93% at
     512/1024/2048/4096. The matched NoPE Transformer reaches
     17.19% +/- 2.71%, 16.15% +/- 6.51%, 22.92% +/- 5.02%, and
     10.94% +/- 2.71%.
   - Small HF-Mamba 8-slot final-query-only baseline, same curriculum and
     three seeds: 18.23% +/- 3.61%, 13.02% +/- 1.80%,
     13.54% +/- 7.86%, and 18.23% +/- 1.80% at 512/1024/2048/4096.
   - Parameter-matched HF-Mamba d192 8-slot final-query-only baseline:
     17.71% +/- 8.02%, 13.54% +/- 6.51%, 19.27% +/- 7.71%, and
     16.15% +/- 6.31% at 512/1024/2048/4096.
   - 6-slot curriculum ablation, seed42 only: direct final-only training
     without the active-slot curriculum reaches 23.44% at 4096, compared with
     95.31% for the curriculum run at the same seed.
   - 8-slot curriculum ablation, seed42 only: direct final-only training
     without the active-slot curriculum reaches 12.50% at 4096, compared with
     98.44% for the curriculum run at the same seed. The direct run's gate
     diagnostic does not separate matched and unmatched value tokens.
   - Scratch GRU early-learning probe, seed42 only: a 2-layer d_model=128 GRU
     trained for 500 steps on the same 6-slot final-only curriculum reaches
     6.25% train accuracy at step 500 and 6.25% eval accuracy at 4096. This is
     not a complete recurrent baseline because the unfused CUDA GRU path is
     much slower than the MOGT/Transformer paths.
   - Gate diagnostic: in a seed-42 2-slot diagnostic run, block 0 opens the
   mean rank-wise gate to 32.64% on matched value tokens, 2.04% on unmatched
   value tokens, and below 0.2% on SET/slot/filler/QUERY tokens.

7. The hybrid LM path is now executable but still preliminary.
   - Evidence: `model_hybrid.py`, `train_budget_hybrid.py`, and
     `benchmark_runs/hybrid_alt_ctx8192_d128_l2_steps10_seed42.json`.
   - One-seed wiring pilot, `ctx=8192`, `d_model=128`, two layers, 10 steps:
     alternating hybrid best val loss is 10.4828, while the matched tiny
     Transformer pilot is 10.5214.
   - First 4-layer ratio pilot, seed42, `ctx=8192`, 5 steps, one validation
     batch: val loss moves from 10.6355 at 0% MOGT layers to 10.5526 at 100%
     MOGT layers. This is a queueing signal for longer sweeps, not a paper
     result.
   - Initialization caveat: when Transformer attention output projections are
     also zero-initialized, the 0% control improves to 10.6032 and the
     25/50/75/100% MOGT points cluster at 10.5540/10.5543/10.5532/10.5526.
     This weakens the original monotone-ratio story and should be reported as
     an initialization-control result.
   - 50-step zero-init follow-up on 0/25/100% MOGT gives val loss
     9.5796/9.5618/9.6249. The first useful LM direction is low-ratio hybrid
     insertion; pure MOGT is currently a negative control at this budget.
   - 3-seed 50-step zero-init sweep on 0/25/50% MOGT gives mean val loss
     9.5865/9.5784/9.6034 with std 0.0091/0.0163/0.0167. This is a weak,
     variance-overlapping signal for 25% insertion, not a publishable LM win.
   - Three-seed 200-step zero-init follow-up with four validation batches gives
     7.6069 +/- 0.0091 for 0% MOGT and 7.4897 +/- 0.0107 for 25% MOGT. This is
     the first clear small-LM positive signal for low-ratio MOGT insertion, but
     it remains a small WikiText-103 pilot, not a general LM claim.
   - Single-MOGT-layer position follow-up at the same budget suggests later
     insertion is better in this tiny LM. Attention-only is
     7.6069 +/- 0.0091; MOGT at layer 1/2/3 gives
     7.4897 +/- 0.0107, 7.4675 +/- 0.0110, and 7.4539 +/- 0.0092.
     Layer 0 has only seed42 so far and gives 7.5189. This is a useful scaling
     target, not a final architectural rule.
   - The first late-layer scale-up keeps the signal alive: at 500 steps with
     eight validation batches, attention-only is 6.8137 +/- 0.0139 while a
     single layer-3 MOGT insertion is 6.7241 +/- 0.0132 across seeds 7/42/123.
     This is still a small WikiText-103 pilot, but it is now a stronger hybrid
     LM candidate than the original 50/200-step probes.
   - A 1000-step follow-up in the same setting gives 6.3868 +/- 0.0098 for
     attention-only and 6.3357 +/- 0.0128 for layer-3 MOGT. The gap narrows
     but persists across three seeds, so the next falsification target is
     scale rather than early-step noise.
   - The first width scale probe also remains positive. At `d_model=192`,
     four layers, 500 steps, and eight validation batches, attention-only is
     6.5024 +/- 0.0141 while layer-3 MOGT is 6.4544 +/- 0.0134 across
     seeds 7/42/123. The layer-3 hybrid has slightly fewer parameters in this
     setup (11.40M vs 11.42M), so this is not a parameter-count artifact in
     MOGT's favor.
   - The same `d_model=192` setup becomes a boundary at 1000 steps:
     attention-only is 6.0997 +/- 0.0126, layer-2 MOGT is
     6.1049 +/- 0.0147, and layer-3 MOGT is 6.1337 +/- 0.0162.
     Layer 2 nearly matches attention-only, while layer 3 clearly trails. This
     rules out the simple claim that a late single MOGT layer monotonically
     improves LM quality as training continues.
   - A two-MOGT-layer middle hybrid (layers 1+2) does not fix the d192/1000
     boundary: it reaches 6.1416 +/- 0.0180, worse than attention-only and
     layer-2-only. This argues against simply increasing MOGT layer count in
     this small LM setting.
   - Learning-rate fairness check: lr=5e-4 improves both models, but the tuned
     attention-only control remains better. At `d_model=192`, 1000 steps, and
     lr=5e-4, attention-only reaches 5.8753 +/- 0.0108 while layer-2 MOGT
     reaches 5.9058 +/- 0.0430. This weakens the hybrid LM superiority story
     and reframes the positive results as early/mid-budget optimization
     signals.
   - Residual-scale diagnostic: the hybrid path now exposes
     `--mogt-residual-scale` and `--mogt-ffn-residual-scale`. In the seed-42
     `d_model=192`, layer-2, 1000-step, lr=5e-4 setting, MOGT readout residual
     scales 0.25/0.5/0.75/1.0 give val loss
     5.9199/5.9110/5.9174/5.9545. The same-seed attention-only control is
     still better at 5.8877, so this is evidence for better fusion/gating,
     not a win.
   - Three-seed fixed scale-0.5 confirmation is now done. At the same
     d192/layer2/1000-step/lr=5e-4 setting, attention-only reaches
     5.8753 +/- 0.0108, while layer-2 MOGT with readout residual scale 0.5
     reaches 5.8775 +/- 0.0293. It beats attention on paired seeds 7 and 123
     and fixes much of the unscaled layer-2 gap (5.9058 +/- 0.0430), but its
     aggregate mean remains 0.0022 worse than attention.
   - Learned residual mixing is implemented and currently negative in its
     naive form. `--mogt-residual-gate` adds a learned scalar gate per MOGT
     block, with `--mogt-residual-gate-init` controlling the initial sigmoid
     value. A one-step smoke run passed, but the seed-42 d192/layer2/lr5e-4
     budget run gives val loss 5.9752, worse than fixed scale 0.5 at 5.9110
     and same-seed attention at 5.8877. Future gate work needs constrained
     dynamics, not just an unconstrained learned scalar.
   - Residual-scale schedules are implemented and currently negative in the
     first budget diagnostic. `--mogt-residual-scale-start` plus
     `--mogt-residual-scale-warmup-steps` linearly warms the MOGT readout
     residual scale to the target. In seed42 d192/layer2/lr=5e-4,
     `0.25 -> 0.5` over 250 steps reaches val loss 5.9497, worse than fixed
     scale 0.5 at 5.9110 and attention at 5.8877.
   - Optimizer partitioning is the first strong positive follow-up at this
     boundary. `--mogt-lr-mult` creates a lower-LR optimizer group for MOGT
     block parameters. In seed42 d192/layer2/lr=5e-4 with fixed residual scale
     0.5, `--mogt-lr-mult 0.5` reaches val loss 5.8601, better than both
     fixed scale at the default multiplier (5.9110) and same-seed attention
     (5.8877). The three-seed confirmation is now positive: attention-only is
     5.8753 +/- 0.0108, while layer-2 MOGT with residual scale 0.5 and
     `mogt_lr_mult=0.5` is 5.8511 +/- 0.0117. It beats attention on all three
     paired seeds, with a mean advantage of -0.0242.
   - First width migration check is neutral. At `d_model=256`, seed42,
     1000 steps, lr=5e-4, attention-only reaches 5.7550 while layer-2 MOGT
     with residual scale 0.5 and `mogt_lr_mult=0.5` reaches 5.7558. This
     suggests the recipe does not collapse with width, but it does not yet
     support a width-scaling win.
   - This only supports "the hybrid route is trainable enough to scale up." It
     does not support a language-modeling superiority claim.

## Claims We Cannot Make Yet

1. "MOGT overturns Transformer."
   - Current evidence says no on short-budget WikiText LM quality:
     scratch Transformer 32k/200-step loss is 6.1009, while the MOGT
     baseline_v1 three-seed mean is 6.4067.

2. "MOGT is generally better than attention."
   - Missing evidence: more datasets, more training budgets, more seeds,
     stronger attention baselines, and clear quality/throughput tradeoff curves.

3. "MOGT has solved long-context retrieval."
   - Current controlled dynamic key/value recall is negative for original MOGT:
     Transformer learns the small task, while MOGT stays near chance.
   - Unconditioned tracked multi-slot state routing is only a weak
     positive/neutral signal: MOGT is modestly better than NoPE Transformer on
     some 2-slot and 4-slot probes, but neither model solves the task.
   - Rank-wise and coupled write-forget gates without prefix addressing improve some long-context
     multi-slot numbers but do not solve selective routing. In the seed-42
     2-slot dense probe, the best coupled rank variant reaches 67.19% at 4096
     but only 54.69%/53.12%/51.56% at 512/1024/2048.
   - Boundary: the new slot-addressed result solves tracked, prefix-conditioned
     2-slot final-only, 4-slot dense-supervision, and 4/6/8-slot final-only
     routing with a slot-count curriculum, not arbitrary retrieval. Direct
     final-only learning without curriculum is still open.
   - Missing evidence: needle/passkey, multi-query recall, and extrapolation
     evaluations with matched parameter counts and matched training budgets.

4. "The Triton implementation is the final systems contribution."
   - Current `triton_hybrid` path is useful, but the paper needs a cleaner
     fused or near-fused implementation story with benchmarked kernels.

5. "We have already beaten all recurrent baselines."
   - Partially addressed. The paper now includes small and parameter-matched
     HF-Mamba 8-slot baselines plus a scratch GRU early-learning probe. It
     still needs a broader optimized recurrent/SSM baseline set such as RWKV,
     RetNet, GLA, or a more heavily tuned Mamba control under matched budget.

6. "Gated MOGT is ready as a general LM architecture."
   - Not yet. The positive result is synthetic state tracking, not WikiText
     perplexity or broad language modeling quality.
   - The new hybrid pilot is encouraging, but it is one seed and 10 steps; it
     must be expanded to real token budgets and layer-ratio sweeps before it
     can carry a paper claim.

## Viable Top-Tier Paper Angle

A defensible first paper should not start from "we replace Transformer
everywhere." A stronger angle is:

```text
MOGT is a matrix-valued affine transport operator with an associative scan
structure, designed for long-context sequence modeling. It offers a different
quality/throughput/memory tradeoff from attention and exposes a path to
hardware-conscious recurrent long-context models.
```

This can become top-tier only if the experiments show at least one crisp win:

1. Long-context efficiency win at comparable quality.
2. Long-context retrieval or state-tracking win at matched budget.
3. Kernel/system scaling win with a credible path to full training.
4. A hybrid architecture win where MOGT replaces some attention layers without
   sacrificing perplexity.

The best current angle is narrower and stronger:

```text
Coupling token-dependent writing and forgetting in matrix-valued affine
transport produces a scan-compatible recurrent operator that learns
overwrite-style state tracking and extrapolates from context 512 to 8192 across
three seeds. Adding prefix-conditioned slot addressing extends the mechanism to
tracked multi-slot state routing in controlled synthetic settings.
```

This wording should be used carefully: the current strongest comparison is
against synthetic state tracking. The coupled write-forget variant beats the
NoPE Transformer baseline on single-slot dense tracking, and the slot-addressed
variant beats NoPE on tracked 2-slot final-only, 4-slot dense, and 4/6/8-slot
final-only-with-curriculum routing; it also beats small and parameter-matched
HF-Mamba baselines on the 8-slot curriculum. The project still needs direct
learning without curriculum, broader recurrent baselines, and language-modeling
or hybrid-attention evidence before making any general Transformer-replacement
claim.

## Evidence Gap Checklist

- Add a fair synthetic long-context benchmark with MOGT and scratch Transformer.
  - Done for key/value recall smoke, modular state tracking, and last-value
    tracking.
- Run smoke tests and save JSON artifacts.
  - Done for recall and last-value tracking.
- Run 3-seed synthetic experiments at increasing context lengths.
  - Done for last-value tracking at train context 128 and eval contexts
    128/256/512/1024.
- Add an affine/state-tracking synthetic task that tests the operator's natural
  recurrence bias, not just generic memorization.
- Re-run WikiText with at least 3 Transformer seeds or label it clearly as a
  one-seed anchor.
- Add throughput/memory curves at 8k, 16k, 32k, 64k, and 128k where possible.
- Add ablations: rank `r`, scan implementation, Cayley vs matrix exponential,
  damping, depth, and MOGT/attention hybrid ratios.
- Add stronger baselines before paper submission: Transformer, Mamba-style SSM,
  RWKV/RetNet/GLA-style linear attention if practical.

## Reviewer Objections To Pre-Answer

1. Is the improvement from the operator, from training tricks, or from the loss?
   - Need ablations and identical training loops.

2. Is the long-context claim only a memory artifact?
   - Initial answer: last-value tracking measures quality at longer contexts,
     not just "fits in memory." Need larger-scale confirmation.

3. Is the synthetic task cherry-picked?
   - Current answer: dynamic key/value recall is negative, last-value tracking
     is positive for gated MOGT. Need at least one more positive synthetic
     family before claiming a broad long-context advantage.

4. Does the method scale beyond toy runs?
   - Need parameter-matched runs and throughput/memory scaling.

5. Are baselines tuned fairly?
   - Need matched params, matched tokens, matched optimizer, matched context,
     and transparent failure/OOM reporting.

## Immediate Next Experiments

1. Smoke-run `benchmark_synthetic_recall.py` for MOGT and Transformer.
   - Done on 2026-05-03. See
     `benchmark_runs/synthetic_recall_summary_smoke.md`.
2. Run controlled last-value tracking.
   - Done on 2026-05-03. See
     `benchmark_runs/synthetic_last_value_summary_20260503.md`.
3. Scale last-value tracking to train contexts 512/1024 and eval contexts up to
   8k/16k.
   - Initial seed-42 train512 stress test was negative for independently gated
     MOGT and positive for NoPE Transformer.
   - Identity transport plus dual gates and dense supervision fixed
     learnability at train512, but not far extrapolation.
   - Coupled write-forget identity transport fixed the single-slot
     far-extrapolation gap through 8192 across 3 seeds.
4. Promote NoPE/position-robust attention to a first-class baseline in every
   synthetic table.
   - Updated on 2026-05-04 for last-value train-context-128: NoPE Transformer
     was rerun with 2048 eval examples per context across seeds 7/42/123.
   - Updated on 2026-05-04 for last-value train-context-512 dense tracking:
     NoPE Transformer was rerun under the standard report schema across seeds
     7/42/123, reaching 78.65% +/- 12.63% at context 8192.
5. Add a second positive recurrent-state task, such as associative stack depth,
   bounded counter with curriculum, or multi-slot last-value tracking.
   - Improved on 2026-05-03: prefix-conditioned slot-addressed coupled MOGT is
     now a strong positive result for tracked 2-slot routing and a seed42
     positive probe for 4-slot routing.
6. Add a gated MOGT language-modeling ablation or hybrid attention/MOGT block
   experiment before making any LM claim.
   - Started on 2026-05-04: initial alternating hybrid model and budget
     training script are in place, with a 10-step `ctx=8192` pilot and a
     5-step 0/25/50/75/100% ratio pilot plus a 50-step 0/25/100% zero-init
     follow-up and a 3-seed 50-step 0/25/50% sweep. A 3-seed 200-step 0/25%
     follow-up is now done; the d192/1000-step checks found that default
     layer-2 MOGT is close but below tuned attention, while a fixed residual
     scale sweep narrows the seed-42 gap and peaks around 0.5. The three-seed
     scale-0.5 confirmation beats attention on 2/3 seeds but remains slightly
     worse in aggregate. Naive learned residual mixing was tested and is worse
     than fixed scale; a simple 0.25-to-0.5 scale warmup is also worse than
     fixed scale in seed42. The lower MOGT learning-rate diagnostic is
     positive across seeds 7/42/123; next step is a slightly larger or longer
     budget confirmation before any broad LM claim. The first d256 check is
     neutral, so larger-width work needs either more seeds, longer budget, or
     an adjusted placement/learning-rate recipe.
