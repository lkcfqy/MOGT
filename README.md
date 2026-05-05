# MOGT / GMAR Research Freeze

This repository is the clean handoff version of the MOGT research project. It
keeps the code, reports, and paper materials needed to resume the work on a new
GPU VM. Large checkpoints, caches, logs, and old exploratory scripts are not
tracked.

## Current Claim

The project is currently scoped as a mechanism paper:

> Coupled write-forget gating and prefix-conditioned slot addressing make a
> scan-compatible matrix-valued recurrent operator solve controlled
> long-context state-tracking tasks where matched NoPE Transformer and
> HF-Mamba baselines struggle.

It does **not** yet claim that MOGT generally replaces Transformer for language
modeling.

## Resume On A New VM

```bash
git clone https://github.com/lkcfqy/MOGT.git
cd MOGT
bash scripts/bootstrap_new_vm.sh
```

The bootstrap script creates `.venv`, installs `requirements-core.txt`,
regenerates paper/report summaries, and validates the stored experiment reports.
Optional CUDA-extension baselines can be installed with:

```bash
INSTALL_OPTIONAL_GPU=1 bash scripts/bootstrap_new_vm.sh
```

## Start Here

- [Project freeze / VM handoff](./docs/PROJECT_FREEZE_20260505.md)
- [New VM quickstart](./docs/NEW_VM_QUICKSTART.md)
- [Top-tier roadmap](./docs/TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md)
- [Claim ledger](./docs/claim_ledger.md)
- [Paper draft](./paper/main.tex)
- [Result snapshot](./paper/results_snapshot.md)

## Strongest Evidence

- Single-slot overwrite tracking:
  - Coupled MOGT: `100.00% +/- 0.00%` through context 8192.
  - NoPE Transformer: `78.65% +/- 12.63%` at context 8192.
- Tracked 4-slot final-query routing with curriculum:
  - Slot-addressed MOGT: `94.27% +/- 2.39%` at context 4096.
  - NoPE Transformer: `21.35% +/- 6.31%`.
  - Parameter-matched HF-Mamba d192: `19.79% +/- 10.97%`.
- Hybrid LM boundary:
  - d192/layer2/residual-scale-0.5/MOGT-LR-mult-0.5 wins a 3-seed small-budget
    WikiText-103 check.
  - d256 seed42 is neutral, so this is not yet a width-scaling claim.

## Core Files

- `affine_scan.py`: associative affine recurrence reference implementations.
- `model_mogt.py`: MOGT/GMAR-style recurrent block and language-model wrapper.
- `model_hybrid.py`: hybrid Transformer + MOGT backbone.
- `triton_scan.py`: current Triton/hybrid scan code paths.
- `chunked_lm_loss.py`: memory-efficient language-modeling loss.
- `train_budget_hybrid.py`: main hybrid LM budget runner.
- `train_budget_transformer.py`: matched scratch Transformer runner.
- `train_budget_baseline.py`: Mamba-style budget baseline runner.
- `benchmark_synthetic_last_value.py`: single-slot state-tracking experiments.
- `benchmark_synthetic_multislot.py`: tracked multi-slot routing experiments.
- `benchmark_backbone_throughput.py`: bounded backbone timing probe.
- `experiment_report.py`: standard report schema helpers.
- `summarize_paper_results.py`: regenerates `paper/results_snapshot.md`.
- `validate_experiment_reports.py`: validates stored JSON reports.

## Preserved Evidence

- `benchmark_runs/`: JSON and Markdown experiment artifacts.
- `paper/`: paper draft, BibTeX, and generated result snapshot.
- `docs/`: roadmap, claim ledger, protocol, and VM handoff notes.

## Not In Git

The following are intentionally excluded:

- `baseline_checkpoints/`
- `mogt_checkpoints/`
- `dataset_cache/`
- `profile_runs*/`
- logs, pid files, model weights, local archives, and virtual environments

This keeps the GitHub repo small and easy to clone.

## Quick Verification

```bash
python3 -m py_compile \
  affine_scan.py model_mogt.py model_hybrid.py train_budget_hybrid.py \
  benchmark_backbone_throughput.py experiment_report.py \
  summarize_paper_results.py summarize_standard_reports.py \
  validate_experiment_reports.py

python3 validate_experiment_reports.py
python3 summarize_paper_results.py
```

Expected freeze-time validation:

```text
checked=118 skipped=153 failures=0
```
