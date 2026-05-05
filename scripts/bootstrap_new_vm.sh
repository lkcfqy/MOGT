#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_OPTIONAL_GPU="${INSTALL_OPTIONAL_GPU:-0}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"

echo "[MOGT] Project root: $ROOT_DIR"
echo "[MOGT] Python: $($PYTHON_BIN --version)"

if [ ! -d "$VENV_DIR" ]; then
  echo "[MOGT] Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "[MOGT] Installing core research dependencies"
python -m pip install -r requirements-core.txt

if [ "$INSTALL_OPTIONAL_GPU" = "1" ]; then
  echo "[MOGT] Installing optional GPU baselines/extensions"
  python -m pip install -r requirements-optional-gpu.txt || {
    echo "[MOGT] Optional GPU dependencies failed. Core repo setup is still usable."
    echo "[MOGT] Re-run later after checking CUDA, PyTorch, and compiler versions."
  }
else
  echo "[MOGT] Skipping optional GPU dependencies. Set INSTALL_OPTIONAL_GPU=1 to install flash-attn/Mamba extras."
fi

if [ "$RUN_VALIDATION" = "1" ]; then
  echo "[MOGT] Running compile smoke checks"
  python -m py_compile \
    affine_scan.py \
    model_mogt.py \
    model_hybrid.py \
    train_budget_hybrid.py \
    benchmark_backbone_throughput.py \
    experiment_report.py \
    summarize_paper_results.py \
    summarize_standard_reports.py \
    validate_experiment_reports.py

  echo "[MOGT] Regenerating report indexes"
  python summarize_standard_reports.py
  python summarize_paper_results.py
  python validate_experiment_reports.py
fi

cat <<'EOF'

[MOGT] Bootstrap complete.

Start reading here:
  docs/PROJECT_FREEZE_20260505.md
  docs/TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md
  paper/main.tex
  paper/results_snapshot.md

Useful next commands:
  source .venv/bin/activate
  python validate_experiment_reports.py
  python summarize_paper_results.py

Paper build on a machine with LaTeX:
  cd paper
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

EOF
