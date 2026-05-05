# New VM Quickstart

This repo is set up so a new GPU VM can resume the MOGT research line without
copying old checkpoints or caches.

## One-Command Bootstrap

```bash
git clone https://github.com/lkcfqy/MOGT.git
cd MOGT
bash scripts/bootstrap_new_vm.sh
```

The script creates `.venv`, installs core dependencies, compiles the main code
paths, regenerates report indexes, and validates existing experiment reports.

Optional CUDA-extension baselines can be installed with:

```bash
INSTALL_OPTIONAL_GPU=1 bash scripts/bootstrap_new_vm.sh
```

Use the optional path only after the CUDA/PyTorch/compiler stack is healthy;
`flash-attn`, `mamba-ssm`, and `causal-conv1d` are useful but not required for
reading the paper or validating the stored reports.

## Resume Points

Start from:

- `docs/PROJECT_FREEZE_20260505.md`
- `docs/TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md`
- `docs/claim_ledger.md`
- `paper/main.tex`
- `paper/results_snapshot.md`

## What Is Not In Git

The GitHub repo intentionally excludes large or disposable runtime state:

- `baseline_checkpoints/`
- `mogt_checkpoints/`
- `dataset_cache/`
- `profile_runs*/`
- `*.log`
- `*.pid`
- `*.pt`, `*.pth`, `*.safetensors`

The preserved evidence is in `benchmark_runs/`, `paper/`, and `docs/`.

## Fast Verification

```bash
source .venv/bin/activate
python validate_experiment_reports.py
python summarize_paper_results.py
```

Expected freeze-time validation:

```text
checked=118 skipped=153 failures=0
```
