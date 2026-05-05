# Draft Paper Outline

## Working Title

Slot-Addressed Coupled Write-Forget Affine Transport for Recurrent State
Tracking

## One-Sentence Claim

Matrix-valued affine transport becomes a useful long-context recurrent operator
when writing and forgetting are coupled; this solves single-slot overwrite
state tracking through 8192 across three seeds, and prefix-conditioned
slot-addressing extends the result to tracked multi-slot routing, including
4-slot, 6-slot, and 8-slot final-query-only learning with a slot-count
curriculum, but it is not yet a general Transformer replacement.

## Abstract Skeleton

Transformers remain strong general-purpose sequence models, but their attention
mechanism couples sequence length to quadratic interaction patterns. We study a
non-attention alternative based on matrix-valued affine transport, where hidden
state evolves through an associative recurrence `H_t = U_t H_{t-1} + V_t`.
The associative structure admits scan-style implementations and gives a natural
path to long-context recurrent computation. In early experiments, however, the
ungated operator fails on overwrite-style memory tasks, revealing that
orthogonal-like transport alone lacks a content-dependent forgetting mechanism.
We introduce a lightweight coupled write-forget gate and show that this variant
learns last-value state tracking and extrapolates beyond its training context.
On a 3-seed synthetic benchmark trained at context 512 with dense supervision,
coupled MOGT reaches 100.00% +/- 0.00% accuracy at every evaluated context from
512 through 8192, compared with 78.65% +/- 12.63% for the current
standard-schema NoPE Transformer rerun at 8192. A seed-42 stress probe remains
at 100% through context 65,536 with fewer eval examples. With
prefix-conditioned slot-addressing, MOGT reaches
86.46% +/- 3.25% at context 4096 on tracked 2-slot routing, compared with
44.79% +/- 9.42% for NoPE Transformer. On tracked 4-slot final-query-only
training with a 2-to-4-slot curriculum, MOGT reaches 94.27% +/- 2.39% at
4096, compared with 21.35% +/- 6.31% for the current standard-schema NoPE
Transformer rerun and 19.79% +/- 10.97% for parameter-matched HF-Mamba d192;
the 6-slot version reaches 96.88% +/- 2.71% versus 21.88% +/- 8.27%, and the
8-slot version
reaches 85.42% +/- 11.93% versus 10.94% +/- 2.71% for NoPE Transformer and
18.23% +/- 1.80% for HF Mamba d128, while parameter-matched HF Mamba d192
reaches 16.15% +/- 6.31%. The result is still synthetic and does not establish
language modeling superiority.

## Intended Contributions

1. Define matrix-valued affine transport as a scan-compatible sequence operator.
2. Show why ungated skew/orthogonal transport is insufficient for overwrite
   memory.
3. Add coupled write-forget gating as a minimal architectural fix.
4. Add prefix-conditioned slot-addressed gate inputs for selective routing.
5. Provide controlled synthetic benchmarks separating key/value retrieval,
   modulo state tracking, and last-value state tracking.
6. Report both positive and negative results against matched Transformer
   baselines, including RoPE and NoPE variants.

## Current Experimental Story

Positive:

- Gated MOGT learns last-value tracking at train context 128 and beats a
  matched RoPE Transformer at 256/512/1024.
- Matched-eval NoPE Transformer is highly competitive on the same train-context
  128 last-value task: with 2048 eval examples per context across three seeds,
  it reaches 68.64% +/- 15.51% at 1024 versus gated MOGT's
  69.71% +/- 18.69%. The paper should use this as evidence that
  position-robust attention is a serious baseline, not as a MOGT win.
- Coupled write-forget MOGT solves the train-context-512 dense single-slot task
  across seeds 7/42/123 and stays at 100% through context 8192. The matched
  standard-schema NoPE Transformer reaches 78.65% +/- 12.63% at 8192.
- A seed-42 long probe stays at 100% through 65,536 with 16 eval examples per
  context.
- Gate diagnostics show the first block opens on value tokens and stays nearly
  closed on filler/SET/QUERY tokens.
- Slot-addressed coupled MOGT solves tracked 2-slot routing much better than
  NoPE Transformer across 3 seeds: 4096 accuracy is 86.46% +/- 3.25% vs
  44.79% +/- 9.42%.
- The 2-slot result survives final-query-only training: 4096 accuracy is
  96.35% +/- 3.25%, compared with 42.71% +/- 21.67% for NoPE Transformer.
- Slot-addressed coupled MOGT also solves the 4-slot probe:
  3-seed 4096 accuracy is 98.96% +/- 0.90%, compared with
  24.48% +/- 7.05% for NoPE Transformer.
- A seed42 dense-to-final curriculum solves 4-slot weak-supervision transfer
  at 4096 with 96.88% accuracy.
- A 2-to-4-slot curriculum solves 4-slot final-query-only training across
  three seeds: 4096 accuracy is 94.27% +/- 2.39%, compared with
  21.35% +/- 6.31% for the current standard-schema NoPE Transformer rerun and
  19.79% +/- 10.97% for parameter-matched HF-Mamba d192.
- A 2-to-6-slot curriculum solves 6-slot final-query-only training across
  three seeds: 4096 accuracy is 96.88% +/- 2.71%, compared with
  21.88% +/- 8.27% for NoPE Transformer.
- A 2-to-8-slot curriculum remains strongly positive but exposes a scaling
  boundary: 4096 accuracy is 85.42% +/- 11.93%, compared with
  10.94% +/- 2.71% for NoPE Transformer, 18.23% +/- 1.80% for HF Mamba d128,
  and 16.15% +/- 6.31% for parameter-matched HF Mamba d192.
- A seed42 direct 6-slot final-only ablation reaches only 23.44% at 4096,
  compared with 95.31% for the curriculum run, so the curriculum should be
  presented as an optimization condition.
- A seed42 direct 8-slot final-only ablation reaches only 12.50% at 4096,
  compared with 98.44% for the curriculum run; its gate diagnostic also fails
  to separate matched from unmatched value tokens.
- A scratch GRU seed42 early-learning probe reaches only 6.25% train accuracy
  at step 500 and 6.25% eval accuracy at 4096, but it is not a complete
  recurrent baseline because the unfused CUDA GRU path is slow.
- Small and parameter-matched HF-Mamba 8-slot baselines are now 3-seed SSM
  controls and remain far below MOGT at 4096.

Negative:

- Original ungated MOGT stays near chance on last-value tracking.
- Original MOGT also fails the dynamic key/value recall learnability probe.
- Binary/modular state tracking did not become a clean positive result.
- Direct train-context-512 favors NoPE Transformer over independently gated
  MOGT.
- Identity transport plus independent dual gates and dense state supervision
  fixes MOGT train512 learnability across 3 seeds, but NoPE Transformer remains
  stronger at far extrapolation for that non-coupled variant.
- Multi-slot selective tracking gives only a weak second-task signal so far:
  unconditioned MOGT is modestly better in some small probes, but neither model
  solves it without prefix-conditioned addressing.
- Rank-wise gates without prefix addressing help some long-context multi-slot
  cases but are not enough to count as a solved selective memory mechanism.
- Direct 4-slot final-only from scratch is not solved yet: in the current
  standard-schema 3-seed rerun, MOGT reaches 44.79% +/- 17.21% at 4096 and
  NoPE reaches 25.00% +/- 5.63%. Four-slot write-only supervision is
  seed-sensitive.
- WikiText short-budget LM quality still favors scratch Transformer.

## Main Figures/Tables To Build

1. Operator diagram: affine transport recurrence and associative scan.
2. Last-value 3-seed table: gated MOGT vs RoPE Transformer.
3. Position baseline table: RoPE vs NoPE Transformer.
4. Ablation table: ungated MOGT, sequential MOGT, gated MOGT.
5. Scaling stress table: train128 long eval, train512 direct/curriculum, and
   coupled write-forget through 65k.
6. Throughput/memory curves for 128 to 8192 synthetic contexts.
7. LM sanity table: WikiText MOGT vs scratch Transformer vs scratch Mamba.

## Experiments Needed Before Submission

1. Repeat last-value NoPE evaluation with the same number of eval examples as
   the main table.
   - Done on 2026-05-04 for train context 128, eval contexts 128/256/512/1024.
     Result: 68.64% +/- 15.51% at 1024, making NoPE competitive with gated
     MOGT on this specific task.
2. Extend slot-addressed coupled write-forget beyond dense synthetic tasks.
   - Current diagnosis: it fixes single-slot far extrapolation, tracked 2-slot
     final-only, tracked 4-slot dense supervision, and tracked 4-slot
     final-only with a slot-count curriculum. Direct 4-slot final-only and
     harder routing remain open.
3. Add harder recurrent-state tasks beyond tracked multi-slot routing.
   - Current tracked 2-slot and 4-slot probes are now strong positives under
     dense supervision, and the 4-slot final-only curriculum is a strong
     positive without dense labels.
   - The 6-slot curriculum probe is now a 3-seed positive; the next step is
     direct training without curriculum or larger slot counts.
   - Direct 6-slot seed42 has now been tested and fails, so direct training
     remains a real open problem rather than an untested assumption.
4. Run rank/gate/damping ablations.
5. Compare against at least one more recurrent/linear baseline if practical.
   - Scratch GRU plus small and parameter-matched HF-Mamba probes are added;
     still need broader or more optimized RWKV/RetNet/GLA/Mamba-style baselines
     if practical.
6. Add a gated MOGT LM or hybrid MOGT-attention experiment.
7. Produce clean throughput and memory scaling plots.

## Submission-Level Claim Boundary

Do not claim "MOGT overturns Transformer." The current evidence supports a
more precise claim:

```text
Coupling token-dependent writing and forgetting is sufficient for affine
transport to solve controlled single-slot overwrite memory and extrapolate far
beyond the training context. Prefix-conditioned slot-addressing extends the
same mechanism to tracked multi-slot routing, including 4-slot final-query-only
and 6/8-slot final-query-only learning with a slot-count curriculum, while
direct harder routing and language modeling remain open.
```

This can become a strong conference submission if the result transfers to at
least one more task family or produces a credible language-modeling/hybrid
attention win.
