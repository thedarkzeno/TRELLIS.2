#!/usr/bin/env bash
# Stage 2 — Dry-run: verify that the training pipeline loads without errors
# Run from repo root: bash scripts/02_tryrun.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
CONFIG="configs/scvae/shape_vae_next_dc_f16c32_fp16_validation.json"
OUTPUT_DIR="results/shape_vae_validation"

echo "==============================="
echo " TRELLIS2 — Training Dry-Run"
echo " Config:     $CONFIG"
echo " Data root:  $ROOT"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --tryrun \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"$ROOT\", \"mesh_dump\": \"$ROOT/mesh_dumps\", \"dual_grid\": \"$ROOT/dual_grid_256\", \"asset_stats\": \"$ROOT/asset_stats\"}}"

echo ""
echo "Dry-run completed successfully — pipeline loads without errors."
