#!/usr/bin/env bash
# Stage 3 — Validation training: 200 steps to confirm full training loop works
# Run from repo root: bash scripts/03_train_validation.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
CONFIG="configs/scvae/shape_vae_next_dc_f16c32_fp16_validation.json"
OUTPUT_DIR="results/shape_vae_validation"

echo "==============================="
echo " TRELLIS2 — Validation Training"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Steps:      200 (validation only)"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Help with VRAM fragmentation on RTX 3090 (24 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"$ROOT\", \"mesh_dump\": \"$ROOT/mesh_dumps\", \"dual_grid\": \"$ROOT/dual_grid_256\", \"asset_stats\": \"$ROOT/asset_stats\"}}"

echo ""
echo "==============================="
echo " Validation training COMPLETE"
echo " Outputs in: $OUTPUT_DIR"
echo "==============================="
echo ""
echo "Checkpoints saved every 100 steps:"
ls -lh "$OUTPUT_DIR/ckpts/" 2>/dev/null || echo "  (no checkpoints yet)"
