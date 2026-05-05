# Repository Structure Standard

Last updated: 2026-05-04

## Purpose

This document defines how the project should be organized while moving from a
research scratchpad to a top-tier-paper codebase.

The current repo contains valuable experimental history. Do not delete or move
large groups of files only for neatness. First make results reproducible, then
move code with import updates and smoke tests in the same change.

## Current Policy

Keep runnable Python files at the repository root for now. This avoids breaking
old benchmark commands and checkpoint/evaluation scripts while the paper story
is still moving.

New strategic documents go in `docs/`.

New generated benchmark reports go in `benchmark_runs/`.

Large checkpoints, caches, logs, and profiler outputs stay out of git.

## Target Layout

The desired long-term layout is:

```text
MOGT/
  README.md
  requirements.txt
  docs/
    TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md
    REPO_STRUCTURE_STANDARD.md
    experiment_protocol.md
    claim_ledger.md
  mogt/
    __init__.py
    affine_scan.py
    chunked_lm_loss.py
    model.py
    hybrid.py
    triton_scan.py
  baselines/
    __init__.py
    transformer.py
    mamba.py
    gru.py
  experiments/
    synthetic/
      last_value.py
      multislot.py
      recall.py
      state_tracking.py
    language_modeling/
      train_mogt.py
      train_transformer.py
      train_hybrid.py
      evaluate_checkpoints.py
      evaluate_hf.py
    systems/
      throughput.py
      profile_train_step.py
  scripts/
    summarize_budget_baselines.py
    summarize_synthetic_last_value.py
    summarize_synthetic_multislot.py
  paper/
    main.tex
    references.bib
  benchmark_runs/
  artifacts/
    logs/
    profile_runs/
    checkpoints/
```

## Migration Order

1. Create import-safe packages:
   - `mogt/`
   - `baselines/`
   - `experiments/`
2. Move core files first:
   - `affine_scan.py`
   - `chunked_lm_loss.py`
   - `model_mogt.py`
   - `triton_scan.py`
3. Update imports and run:
   - `python3 -m compileall -q .`
   - `python3 sanity_affine_scan.py`
   - `python3 sanity_triton_gradients.py`
   - `python3 sanity_triton_training.py`
4. Move baselines.
5. Move experiment scripts.
6. Convert legacy root scripts into thin compatibility wrappers when useful.

## What To Archive

Move to an archive directory only after the current paper tables no longer
depend on them:

- old exploratory passkey/perplexity/lifelong/scaling scripts
- old one-off profile folders
- old train logs
- outdated PDFs

Do not delete benchmark JSONs used by paper tables.

## File Naming Rules

Use descriptive experiment names:

```text
synthetic_multislot8_mogt_slotaddr_coupled_rank_slotcurr_finalonly_ctx512_seed42_steps5000.json
```

Every benchmark artifact name should encode:

- task
- model
- key variant
- train context
- seed
- steps

## Report Rules

Every report should include:

- exact command
- git commit if available
- dataset and tokenizer
- model parameter count
- seed
- train/eval context
- train steps
- eval examples or batches
- wall-clock
- peak memory
- status
- failure reason if failed

## Cleanup Rules

Safe to delete:

- temporary `.pid` files
- obsolete local logs after their metrics are captured in JSON/Markdown
- generated PDFs that can be regenerated
- `__pycache__/`

Do not delete without replacing:

- benchmark JSONs referenced by `docs/claim_ledger.md`
- checkpoint metadata used by `evaluate_checkpoints.py`
- synthetic summaries used by the paper draft
- sanity scripts

## Current Minimal Standard

For the next phase, this is enough:

- Keep root scripts runnable.
- Keep strategic docs in `docs/`.
- Keep current paper draft in `paper/`.
- Keep benchmark artifacts in `benchmark_runs/`.
- Record all new experiments with command, seed, and status.
