#!/usr/bin/env bash
# Stage 6 — Shape Flow model validation training (200 steps)
# Prerequisites:
#   - 04_encode_shape_latents.sh must have been run (shape_latents/shape_enc_next_dc_f16c32_fp16_512 exists)
#   - 05_render_cond.sh must have been run (renders_cond/ exists)
# Run from repo root: bash scripts/06_train_shape_flow.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SHAPE_LATENT_DIR="$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
RENDER_COND_DIR="$ROOT/renders_cond"
CONFIG="configs/gen/slat_flow_img2shape_validation.json"
OUTPUT_DIR="results/slat_flow_shape_validation"

echo "==============================="
echo " TRELLIS2 — Shape Flow Training"
echo " Config:       $CONFIG"
echo " Output:       $OUTPUT_DIR"
echo " Steps:        200 (validation only)"
echo " Model:        mini (256ch, 4 blocks) — pipeline validation only"
echo "==============================="
echo ""
echo "NOTE: On first run, DINOv3 ViT-L/16 will be downloaded from HuggingFace (~1.2 GB)"
echo "      and the TRELLIS.2-4B shape decoder will be downloaded (~3-4 GB) for sampling."
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Help with VRAM fragmentation on RTX 3090 (24 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"$ROOT\", \"shape_latent\": \"$SHAPE_LATENT_DIR\", \"render_cond\": \"$RENDER_COND_DIR\"}}"

echo ""
echo "==============================="
echo " Shape Flow training COMPLETE"
echo " Outputs in: $OUTPUT_DIR"
echo "==============================="
echo ""
echo "Checkpoints saved every 100 steps:"
ls -lh "$OUTPUT_DIR/ckpts/" 2>/dev/null || echo "  (no checkpoints yet)"
