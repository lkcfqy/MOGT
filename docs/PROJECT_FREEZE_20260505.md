# MOGT Project Freeze: 2026-05-05

This document records the fast-close plan for the current L4 VM window. The
project is now frozen around a scoped mechanism paper, not a broad Transformer
replacement claim.

## Current Stage

MOGT is past the idea/prototype-only stage. The repo contains runnable model
code, baseline code, experiment reports, summarizers, a paper draft, and an
honesty ledger. The current paper can be completed as a mechanism paper about
scan-compatible matrix-valued affine recurrence for persistent state tracking.

The strongest supported claim is:

> Coupled write-forget gating and prefix-conditioned slot addressing make a
> scan-compatible matrix-valued recurrent operator solve controlled
> long-context state-tracking tasks where matched NoPE Transformer and
> HF-Mamba baselines struggle.

The strongest unsupported claim is:

> MOGT generally replaces Transformer for language modeling.

## Evidence Snapshot

- Standard report validation: `checked=118 skipped=153 failures=0`.
- Single-slot overwrite tracking:
  - Coupled MOGT: `100.00% +/- 0.00%` through context 8192.
  - NoPE Transformer: `78.65% +/- 12.63%` at context 8192.
- Tracked 4-slot final-query routing with curriculum:
  - Slot-addressed MOGT: `94.27% +/- 2.39%` at context 4096.
  - NoPE Transformer: `21.35% +/- 6.31%`.
  - Parameter-matched HF-Mamba d192: `19.79% +/- 10.97%`.
- Hybrid LM strongest positive:
  - `d_model=192`, layer-2 MOGT, residual scale `0.5`, MOGT LR multiplier
    `0.5`, 1000 steps, context 8192.
  - Attention-only: `5.8753 +/- 0.0108`.
  - Hybrid: `5.8511 +/- 0.0117`.
  - Paired seeds: hybrid wins `3/3`.
- Hybrid LM scale boundary:
  - `d_model=256`, seed 42, same budget.
  - Attention-only: `5.7550`.
  - Hybrid: `5.7558`.
  - Interpretation: neutral, not a width-scaling win.
- Systems boundary:
  - Core affine scan is faster than attention core at 16k/32k in the L4 probe.
  - Backbone forward crosses over only at 32k.
  - This is not yet an end-to-end training or decoding systems result.

## Minimum Viable Paper

The fastest credible paper version should be titled around GMAR/MOGT as a
mechanism:

> Gated Matrix-Valued Associative Recurrence for Long-Context State Tracking

Core sections:

1. Affine recurrence and associative scan law.
2. Failure of raw transport on overwrite memory.
3. Coupled write-forget gate.
4. Prefix-conditioned slot addressing.
5. Single-slot and tracked multi-slot state-tracking results.
6. Baselines: NoPE Transformer, RoPE Transformer where relevant, HF-Mamba d192,
   and GRU early probe as a limited recurrent control.
7. Boundary results: hybrid LM is promising but not conclusive; systems timing
   is preliminary.
8. Limitations and claim boundaries.

This is a credible arXiv/workshop-quality package now and can become a main
track submission after more baseline/ablation work.

## Fast-Close Checklist

Before leaving the VM, preserve these files/directories:

- `paper/`
- `docs/`
- `benchmark_runs/`
- current root Python entrypoints used by the paper/evidence path
- `requirements-core.txt`
- `requirements-optional-gpu.txt`
- `.gitignore`
- `README.md`

Large directories are not needed for the paper freeze unless exact checkpoint
resume is required:

- `baseline_checkpoints/` is about 24G.
- `mogt_checkpoints/` is about 33G.
- `dataset_cache/` is about 2G.
- `profile_runs_*` are useful but not essential because summarized timing
  results are already in `benchmark_runs/` and `paper/results_snapshot.md`.

## Repro Commands

Core sanity and report validation:

```bash
python3 -m py_compile affine_scan.py model_mogt.py model_hybrid.py train_budget_hybrid.py benchmark_backbone_throughput.py experiment_report.py summarize_paper_results.py summarize_standard_reports.py validate_experiment_reports.py
python3 validate_experiment_reports.py
python3 summarize_standard_reports.py
python3 summarize_paper_results.py
```

Paper build, on a machine with LaTeX installed:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## What To Do If Only One More Day Is Available

1. Do not run a broad new sweep.
2. Freeze the paper around synthetic state tracking plus honest LM boundaries.
3. Add final figures/tables from `paper/results_snapshot.md`.
4. Run the validation commands above.
5. Produce the light source archive.

## What To Do If One More Week Is Available

1. Add the most important ablations to isolate the mechanism:
   - coupled vs value-only vs forget-only,
   - prefix addressing vs current-only gate input,
   - curriculum vs direct final-only.
2. Add one more credible recurrent/linear baseline if it can run cleanly.
3. Run d256 hybrid LM seeds 7 and 123 only if the VM has time; otherwise leave
   the current d256 seed42 result as a clearly labeled neutral boundary.
4. Convert the draft into camera-ready style: figures, tighter abstract,
   shorter limitations, and clearer method diagrams.

## Decision

For this VM window, the right move is to ship a truthful mechanism paper package.
The route to a stronger Transformer-disruption paper remains open, but it
requires more LM scale evidence or a real systems kernel story.
