# Top-Tier Roadmap: From MOGT Prototype to Transformer-Disrupting Paper

Last updated: 2026-05-05

## North Star

The goal is not to claim that MOGT already beats Transformer everywhere. The
goal is to produce a top-tier paper that makes a defensible, hard-to-ignore
claim:

> Transformer is not the final form of long-context sequence modeling. For
> tasks that require persistent, updateable state, a scan-compatible
> matrix-valued recurrent operator with write-aligned forgetting and
> prefix-conditioned addressing can beat attention baselines under matched
> budget, while exposing a path to lower-memory long-context models.

This is the first credible route to "disrupting Transformer": identify a
regime where attention has the wrong inductive bias or the wrong hardware
economics, win there cleanly, then expand into hybrid language modeling.

## Paper Thesis

Use a narrower, stronger thesis:

> Matrix-valued affine recurrence is a scan-compatible alternative to attention.
> Raw transport is not enough: overwrite memory requires token-dependent writing
> coupled to forgetting, and selective routing requires prefix-conditioned
> addressing. With these ingredients, MOGT/GMAR solves controlled long-context
> state-tracking tasks that strong attention and SSM baselines struggle with.

Recommended paper name family:

- `GMAR`: Gated Matrix-Valued Associative Recurrence
- `MOGT`: keep as the implementation/project name
- Avoid `HMAR` unless "holonomic" is rigorously defined; avoid "full
  Transformer replacement" until LM evidence exists.

## Non-Negotiable Claim Boundaries

Allowed claims now:

- We implement a real non-attention affine recurrence:
  `H_t = U_t H_{t-1} + V_t`.
- The recurrence has an associative scan structure.
- Coupled write-forget gating fixes a real overwrite-memory failure mode.
- Prefix-conditioned addressing gives strong controlled multi-slot tracking
  results.
- The current system can train long contexts with memory-efficient loss and
  hybrid Triton scan.

Forbidden claims until new evidence exists:

- "MOGT generally beats Transformer."
- "MOGT has solved language modeling."
- "The Triton implementation is the final systems contribution."
- "Lifelong learning / topology / thermodynamics are validated."
- "Synthetic wins imply AGI-scale behavior."

## Target Venues

Primary target: NeurIPS / ICLR main track as an algorithmic-mechanistic paper.

Secondary target: MLSys if the kernel becomes the main contribution. For MLSys,
the bar shifts toward fused kernels, roofline analysis, memory traffic, and
end-to-end throughput.

Fallback / bridge target: a strong workshop or arXiv release if the synthetic
story is strong but LM/hybrid evidence remains incomplete.

## Winning Strategy

### Strategy A: Mechanism Paper

This is the most realistic top-tier route.

Core narrative:

1. Define matrix-valued affine recurrence and its scan semigroup.
2. Show raw transport fails on overwrite/state-tracking tasks.
3. Derive or motivate coupled write-forget gates.
4. Add prefix-conditioned slot addressing for selective state routing.
5. Beat NoPE Transformer, RoPE Transformer, Mamba-family, and GRU/RNN baselines
   on controlled state-tracking tasks under matched budget.
6. Report WikiText and retrieval negatives honestly.

Submission bar:

- At least two task families, not one.
- 3 to 5 seeds for every main table.
- Strong baselines: NoPE Transformer, RoPE Transformer, Mamba/SSM, GRU/RNN, and
  at least one more recurrent/linear baseline if practical.
- Ablations that isolate the mechanism.

### Strategy B: Hybrid LM Paper

This is the route to a stronger "Transformer disruption" claim.

Core narrative:

1. Attention is excellent for content addressing.
2. MOGT/GMAR is better suited for persistent state and long-context compression.
3. A hybrid attention + MOGT model reaches comparable LM loss with better
   long-context memory or better memory/throughput.

Submission bar:

- WikiText/OpenWebText-style LM runs beyond 200 steps.
- Matched parameter count and matched token budget.
- Hybrid ratios: 0%, 25%, 50%, 75%, 100% MOGT layers.
- Long-context eval: passkey/needle, state tracking, perplexity at multiple
  contexts.

### Strategy C: Systems Paper

Only choose this if kernel work becomes dominant.

Core narrative:

1. Matrix-valued affine recurrence can be implemented as a hardware-conscious
   scan primitive.
2. The kernel reduces memory traffic and improves long-context throughput.
3. The model preserves acceptable quality while changing the quality/speed
   frontier.

Submission bar:

- Fused or near-fused affine scan, not only `triton_hybrid`.
- Roofline analysis and measured arithmetic intensity.
- End-to-end training and inference throughput at 8k, 16k, 32k, 64k, possibly
  128k.
- Comparison against FlashAttention/SDPA Transformer and Mamba-style baselines.

## Project Phases

### Phase 0: Freeze the Story

Goal: stop the project from splitting into five papers at once.

Deliverables:

- Keep this roadmap as the authority for next steps.
- Remove older grand-theory text from the clean handoff; only paper-relevant
  claims should remain in the public repo.
- Keep `docs/claim_ledger.md` as the honesty ledger.
- Main paper claim becomes: gated matrix-valued affine recurrence for
  long-context state tracking.

Exit criteria:

- Every README/paper statement agrees with the claim boundary.
- No benchmark table presents proxy transport as full affine recurrence.

### Phase 1: Reproducible Experiment Harness

Goal: make results hard to dismiss.

Deliverables:

- One JSON schema for every experiment:
  - model name
  - parameter count
  - train context
  - eval contexts
  - seed
  - steps
  - batch size
  - tokens seen
  - optimizer
  - peak memory
  - wall-clock
  - accuracy/loss/PPL
  - status: ok / failed / oom / skipped
- One summarizer per experiment family.
- One command manifest listing exact commands for main tables.
- Keep failed/OOM runs in reports; do not hide them.
- Current protocol: `docs/experiment_protocol.md`.
- Current command list: `docs/COMMAND_MANIFEST.md`.

Exit criteria:

- A fresh clone with data/cache available can regenerate the main synthetic
  tables from commands.

### Phase 2: Mechanism Ablations

Goal: prove the win comes from the proposed mechanism, not luck.

Required ablations:

- transport:
  - identity
  - Cayley
  - matrix_exp
- gates:
  - no gate
  - value gate only
  - forget gate only
  - independent value + forget
  - coupled write-forget
  - rank-wise coupled gates
- addressing:
  - current token only
  - current + previous
  - current + prefix
  - current + previous + prefix
- supervision:
  - dense
  - final-only
  - write-only
  - curriculum
  - direct no-curriculum
- architecture:
  - rank `r`
  - depth
  - hidden size
  - scan implementation

Exit criteria:

- The paper can answer: "Why does this architecture work, and which component
  is necessary?"

### Phase 3: Baseline Gauntlet

Goal: remove the "weak baseline" objection.

Baseline set:

- Transformer RoPE
- Transformer NoPE
- Transformer ALiBi or another position-robust variant if practical
- Mamba/HF-Mamba
- parameter-matched Mamba
- GRU or LSTM with a fused/optimized path if available
- RWKV/RetNet/GLA-style baseline if practical

Fairness rules:

- Matched training steps and token budgets.
- Matched eval examples.
- Matched parameter counts where possible.
- Same optimizer defaults unless a baseline has a documented standard setup.
- Report peak memory and wall-clock, not only accuracy.

Exit criteria:

- Main synthetic result remains positive against NoPE Transformer and at least
  one credible SSM/recurrent baseline across 3+ seeds.

### Phase 4: Add a Second Task Family

Goal: avoid "you cherry-picked last-value tracking."

Candidate families:

- bounded counter with distractors
- stack or queue depth tracking
- finite-state automaton / regular-language state tracking
- multi-query dynamic key/value state tracking
- associative state machine with controlled writes

Selection rule:

- Choose the task that exposes persistent updateable state, not arbitrary
  content retrieval.
- Include at least one negative control where MOGT should not win.

Exit criteria:

- MOGT/GMAR wins on at least two distinct state-tracking families.

### Phase 5: Hybrid Language Modeling

Goal: begin the true Transformer-disruption path.

Experiments:

- Build a hybrid model with alternating attention and MOGT blocks.
- Ratios:
  - 0% MOGT: pure Transformer
  - 25% MOGT
  - 50% MOGT
  - 75% MOGT
  - 100% MOGT
- Train on the same token stream and token budget.
- Evaluate:
  - WikiText validation PPL
  - context extrapolation PPL
  - synthetic state tracking
  - passkey/needle as secondary, not primary
  - memory and throughput

Exit criteria:

- A hybrid model matches or nearly matches Transformer PPL while improving
  state tracking or long-context memory/throughput.

Current status:

- Initial alternating hybrid backbone and causal LM wrapper were added on
  2026-05-04 (`model_hybrid.py`, `train_budget_hybrid.py`).
- A one-seed, 10-step WikiText-103 wiring pilot at `ctx=8192`, `d_model=128`,
  and two layers produced best val loss `10.4828` for alternating hybrid vs
  `10.5214` for the same-size Transformer. This only proves the path is
  trainable enough to justify larger layer-ratio sweeps; it is not a quality
  claim.
- The first 4-layer ratio pilot is complete for 0/25/50/75/100% MOGT layers at
  `ctx=8192`, `d_model=128`, 5 steps, seed 42. Validation loss decreases from
  `10.6355` at 0% to `10.5526` at 100%, but the run uses only one validation
  batch and should be treated as a queueing signal for larger sweeps.
- A matched initialization caveat was added immediately after: with
  `--zero-init-attention-out`, the 0% attention-only control improves to
  `10.6032`, while 25/50/75/100% MOGT layers cluster around
  `10.5540 / 10.5543 / 10.5532 / 10.5526`. This means the original monotone
  trend partly reflected residual-branch initialization, not only architecture.
- A slightly longer zero-init control at 50 steps was run for 0/25/100% MOGT.
  Validation loss was `9.5796 / 9.5618 / 9.6249`. The current LM signal points
  toward low-ratio hybrid insertion, not pure MOGT replacement.
- The first 3-seed 50-step zero-init sweep for 0/25/50% MOGT has mean val loss
  `9.5865 / 9.5784 / 9.6034` with std `0.0091 / 0.0163 / 0.0167`. This is a
  weak positive signal for 25% insertion and a negative signal for 50% at this
  budget; it is not statistically strong enough for a paper claim.
- The 200-step, 4-validation-batch follow-up has now been replicated across
  seeds 7/42/123. 0% vs 25% MOGT val loss is `7.6069 +/- 0.0091` vs
  `7.4897 +/- 0.0107`. This is the first clear small-LM positive signal for
  low-ratio MOGT insertion, though still far from a broad LM claim.
- A same-budget single-MOGT-layer position ablation now suggests the useful
  insertion point is later in the 4-layer stack. Attention-only is
  `7.6069 +/- 0.0091`; layer1/2/3 MOGT insertion gives
  `7.4897 +/- 0.0107`, `7.4675 +/- 0.0110`, and
  `7.4539 +/- 0.0092`. Layer0 is only a seed42 diagnostic so far
  (`7.5189`). This should guide the next scale-up, not be treated as a final
  LM architecture rule.
- The first late-layer scale-up is complete: at `500 steps` with eight
  validation batches, attention-only is `6.8137 +/- 0.0139`, while a single
  layer3 MOGT insertion is `6.7241 +/- 0.0132` across seeds 7/42/123. This
  strengthens the hybrid route, but the result is still small-scale
  WikiText-103 evidence.
- A `1000 steps` continuation in the same setup gives attention-only
  `6.3868 +/- 0.0098` and layer3 MOGT `6.3357 +/- 0.0128`. The advantage is
  smaller than at 500 steps but survives the longer budget, so the hybrid route
  is now worth testing at larger model scale.
- First width scale probe: at `d_model=192`, four layers, `500 steps`, and
  eight validation batches, attention-only is `6.5024 +/- 0.0141`, while
  layer3 MOGT is `6.4544 +/- 0.0134`. This is the first cross-width
  replication of the hybrid LM signal.
- Important boundary: the same `d_model=192` setup at `1000 steps` reverses.
  Attention-only reaches `6.0997 +/- 0.0126`; layer2 MOGT nearly matches at
  `6.1049 +/- 0.0147`, while layer3 MOGT trails at
  `6.1337 +/- 0.0162`. A two-MOGT-layer middle hybrid (layers 1+2) also trails
  at `6.1416 +/- 0.0180`. The current hybrid benefit is therefore an early/mid
  budget signal, not yet a durable LM scaling law.
- Learning-rate fairness check: `lr=5e-4` improves the whole setting, but the
  tuned attention-only control is still better: `5.8753 +/- 0.0108` vs
  `5.9058 +/- 0.0430` for layer2 MOGT. This closes the current simple hybrid
  LM win and pushes the paper route back toward mechanism/synthetic wins plus
  cautious hybrid exploration.
- Residual-scale diagnostic has started. The code now supports
  `--mogt-residual-scale` and `--mogt-ffn-residual-scale`; in the seed-42
  `d_model=192`, layer2, 1000-step, lr=5e-4 setting, readout residual scales
  `0.25/0.5/0.75/1.0` produce val loss
  `5.9199 / 5.9110 / 5.9174 / 5.9545`, while same-seed attention-only remains
  better at `5.8877`. Interpretation: the d192 failure is partly a
  fusion/optimization problem, with the best fixed scale near `0.5`, but it is
  not solved.
- The tight three-seed scale-0.5 diagnostic is complete. At the same
  `d_model=192`, layer2, 1000-step, lr=5e-4 setting, attention-only is
  `5.8753 +/- 0.0108`; layer2 MOGT with fixed readout residual scale `0.5` is
  `5.8775 +/- 0.0293`. The scaled hybrid beats attention on paired seeds `7`
  and `123`, and is much better than unscaled layer2 MOGT
  (`5.9058 +/- 0.0430`), but the aggregate mean remains `0.0022` worse than
  attention. Interpretation: residual scaling is a real stabilization lever,
  not yet a Transformer win.
- Learned residual mixing has now been added and falsified in its naive form:
  `--mogt-residual-gate` creates a learned scalar gate on each MOGT readout
  residual branch, initialized by `--mogt-residual-gate-init`. A one-step smoke
  run passes, but the full seed-42 d192/layer2/lr5e-4 diagnostic gives
  `5.9752`, worse than fixed scale `0.5` at `5.9110` and same-seed attention
  at `5.8877`. If this route continues, use constrained gate dynamics such as
  lower gate learning rate, regularization toward the initialization, delayed
  unfreezing, or a bounded residual schedule.
- A bounded residual schedule has now been tested in its simplest form and is
  negative. The code supports `--mogt-residual-scale-start` and
  `--mogt-residual-scale-warmup-steps`, but seed42 d192/layer2/lr=5e-4 with
  `0.25 -> 0.5` over `250 steps` reaches `5.9497`, worse than fixed scale
  `0.5` at `5.9110`. Do not spend a broad sweep on simple linear scale warmup
  until a stronger mechanism motivates it.
- Optimizer partitioning is now the strongest LM follow-up. The code supports
  `--mogt-lr-mult`, which applies a learning-rate multiplier to parameters
  inside MOGT blocks. In seed42 d192/layer2/lr=5e-4 with fixed residual scale
  `0.5`, `--mogt-lr-mult 0.5` reaches `5.8601`, beating same-seed attention
  at `5.8877` and default-MOGT-LR fixed scale at `5.9110`. The three-seed
  confirmation is now positive: attention-only is `5.8753 +/- 0.0108`,
  layer2 MOGT with residual scale `0.5` and `mogt_lr_mult=0.5` is
  `5.8511 +/- 0.0117`, and the hybrid wins on all paired seeds. This upgrades
  the LM story from "close but not won" to "small-budget hybrid win needing
  scale confirmation."
- The first width migration check is neutral rather than positive. At
  `d_model=256`, seed42, 1000 steps, lr=5e-4, the attention-only control
  reaches `5.7550`, while layer2 MOGT with residual scale `0.5` and
  `mogt_lr_mult=0.5` reaches `5.7558`. This means the d192 recipe does not
  collapse at wider model width, but it also does not establish a width-scaling
  win.

### Phase 6: Kernel and Systems Scaling

Goal: turn the scan structure into actual hardware advantage.

Tasks:

- Replace `triton_hybrid` with a cleaner fused or near-fused affine scan.
- Reduce or fuse `carry_apply`.
- Benchmark `matrix_exp` vs `cayley` vs identity transport.
- Profile memory traffic and wall-clock by operator.
- Report 8k/16k/32k/64k and optional 128k.

Exit criteria:

- The system story is no longer "the math could be fast"; it is measured.

## Recommended Directory Standard

Do not move code immediately unless imports are updated in the same commit. The
current root scripts are part of the evidence trail. Use this intended layout
for future cleanup:

```text
MOGT/
  docs/
    TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md
    experiment_protocol.md
    claim_ledger.md
  mogt/
    affine_scan.py
    triton_scan.py
    model_mogt.py
    chunked_lm_loss.py
  baselines/
    model_baseline_transformer.py
    model_baseline_hf_mamba.py
    model_baseline_gru.py
  experiments/
    synthetic/
    language_modeling/
    systems/
  scripts/
    train/
    evaluate/
    summarize/
  paper/
    main.tex
    references.bib
  artifacts/
    benchmark_runs/
    profile_runs/
    logs/
    checkpoints/
```

For now:

- Keep runnable scripts at root.
- Put new strategy and protocols in `docs/`.
- Put future generated reports under `benchmark_runs/`.
- Keep large checkpoints, caches, and logs out of git.

## 30-Day Execution Plan

Week 1:

- Freeze paper name and claim.
- Normalize experiment JSON schema.
- Make one command manifest for current synthetic tables.
- Re-run NoPE Transformer with exactly matched eval examples for last-value
  tracking.
  - Status: done for train-context-128 seeds 7/42/123 on 2026-05-04. The result
    is competitive with gated MOGT at 1024, so future claims must treat NoPE as
    a first-class baseline.
- Standardize the train-context-512 single-slot table.
  - Status: done on 2026-05-04 for coupled write-forget MOGT and NoPE
    Transformer across seeds 7/42/123. Coupled MOGT reaches
    100.00% +/- 0.00% through context 8192; NoPE reaches
    78.65% +/- 12.63% at 8192 under the current standard-schema rerun.
- Standardize one second-task-family table.
  - Status: done on 2026-05-04 for tracked 4-slot final-query-only routing
    with a 2-to-4 slot curriculum. Slot-addressed coupled MOGT reaches
    94.27% +/- 2.39% at context 4096; NoPE reaches 21.35% +/- 6.31%;
    parameter-matched HF-Mamba d192 reaches 19.79% +/- 10.97%.
  - Direct/no-curriculum standard ablation is also done across three seeds:
    MOGT reaches 44.79% +/- 17.21% at context 4096 and NoPE reaches
    25.00% +/- 5.63%, supporting the claim that the slot-count curriculum is
    an optimization condition.
- Add a bounded systems snapshot.
  - Status: done on 2026-05-04 for core operator timing at
    `d_model=768`, rank 16, batch size 1, on NVIDIA L4. The affine
    `triton_hybrid` scan core is slower than attention core at 8k, but faster
    at 16k and 32k. This is only a core-operator result, not an end-to-end
    throughput claim.
  - Backbone-level forward timing is also done for 2-layer `d_model=768`
    identity coupled MOGT vs NoPE Transformer. MOGT is slower at 8k, close at
    16k, and faster at 32k. This still excludes LM head, loss, backward pass,
    optimizer, and KV-cache decode behavior.

Week 2:

- Run missing ablations for coupled write-forget and prefix addressing.
- Add at least one stronger recurrent/linear baseline if practical.
- Produce one consolidated synthetic paper table from scripts.
- Expand the hybrid LM pilot into a layer-ratio sweep:
  0%, 25%, 50%, 75%, and 100% MOGT layers under matched context, token budget,
  and seed set.
- Prioritize 0% vs 25% vs 50% under the zero-init attention control for the
  next 3-seed run; 100% MOGT is currently a negative control after 50 steps.
- Next hybrid step: do not claim LM superiority. Residual scaling is now
  implemented, and the three-seed scale-0.5 confirmation is close but still
  loses in aggregate. Naive learned residual mixing is negative, and the first
  fixed-scale schedule is also negative. The lower MOGT learning-rate
  diagnostic is positive across seeds `7/42/123`, so prioritize a slightly
  larger or longer-budget confirmation with `mogt_lr_mult=0.5` before another
  broad LM sweep. Because the first d256 check is neutral, use more seeds or a
  longer budget before changing the central paper claim.

Week 3:

- Add second task family.
- Run 3 seeds for MOGT, NoPE Transformer, and one SSM/recurrent baseline.
- Write the mechanism section around failure -> gate -> addressing.

Week 4:

- Turn the initial hybrid prototype into either a serious LM section or an
  explicitly deferred follow-up, depending on the layer-ratio sweep.
- Produce draft paper figures.
- Run sanity/repro checks.
- Decide target: NeurIPS/ICLR mechanism paper vs MLSys systems paper.

## Go / No-Go Gates

Go for top-tier main-track submission if:

- Two task families show strong 3+ seed wins.
- NoPE Transformer is not the only baseline.
- Ablations prove coupled write-forget and prefix addressing are necessary.
- WikiText negative results are reported honestly.
- The paper has a clean, reproducible command path.

Do not submit as a broad Transformer replacement paper unless:

- MOGT or a hybrid matches Transformer LM quality under matched token budget, or
- MOGT achieves a decisive long-context efficiency-quality tradeoff with a real
  kernel and strong benchmarks.

## Reviewer Attack List

Prepare answers for:

- Is this just a synthetic trick?
- Is NoPE Transformer tuned fairly?
- Does curriculum hide optimization weakness?
- Is identity transport doing all the work?
- Why does Cayley help LM but identity helps overwrite memory?
- Does the scan implementation matter for quality?
- Does it scale beyond 2-layer toy models?
- Where is the language-modeling evidence?
- Where is the hardware evidence?

## Final Direction

The route to disruption is staged:

1. Win state tracking.
2. Explain the mechanism.
3. Beat strong baselines under matched budget.
4. Add hybrid LM evidence.
5. Turn the scan into measured systems advantage.

Transformer is too strong to overthrow by rhetoric. The viable path is to make
one of its blind spots undeniable, then widen the crack.
