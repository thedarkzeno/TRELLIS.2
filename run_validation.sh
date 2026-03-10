#!/usr/bin/env bash
# Master script — Full validation pipeline for TRELLIS2
# Runs all stages sequentially: dataset → dry-run → training
#
# Usage (from repo root):
#   bash run_validation.sh [--skip-dataset] [--skip-tryrun]
#
# Options:
#   --skip-dataset   Skip data download/processing (use if already prepared)
#   --skip-tryrun    Skip dry-run check and go straight to training
set -e

SKIP_DATASET=false
SKIP_TRYRUN=false

for arg in "$@"; do
    case $arg in
        --skip-dataset) SKIP_DATASET=true ;;
        --skip-tryrun)  SKIP_TRYRUN=true  ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "================================================="
echo " TRELLIS2 — Full Validation Pipeline"
echo " Repo:    $REPO_ROOT"
echo " Skip dataset: $SKIP_DATASET"
echo " Skip tryrun:  $SKIP_TRYRUN"
echo "================================================="

# --- Stage 1: Dataset ---
if [ "$SKIP_DATASET" = false ]; then
    echo ""
    echo ">>> STAGE 1: Dataset preparation"
    bash scripts/01_build_dataset.sh
else
    echo ""
    echo ">>> STAGE 1: Skipped (--skip-dataset)"
fi

# --- Stage 2: Dry-run ---
if [ "$SKIP_TRYRUN" = false ]; then
    echo ""
    echo ">>> STAGE 2: Training dry-run"
    bash scripts/02_tryrun.sh
else
    echo ""
    echo ">>> STAGE 2: Skipped (--skip-tryrun)"
fi

# --- Stage 3: Validation training ---
echo ""
echo ">>> STAGE 3: Validation training (200 steps)"
bash scripts/03_train_validation.sh

echo ""
echo "================================================="
echo " All stages completed successfully!"
echo "================================================="
