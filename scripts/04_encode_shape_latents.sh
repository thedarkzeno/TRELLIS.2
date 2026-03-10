#!/usr/bin/env bash
# Stage 4 — Encode shape latents using pretrained TRELLIS.2-4B encoder
# Prerequisites: 01_build_dataset.sh must have been run (mesh_dumps + dual_grid_256 exist)
# Run from repo root: bash scripts/04_encode_shape_latents.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"
RESOLUTION=512   # O-Voxel resolution to use for encoding (generates dual_grid_512 first)

# Pretrained encoder from TRELLIS.2-4B (downloaded automatically from HuggingFace)
# Requires: huggingface-cli login  (or HF_TOKEN env var)
ENC_PRETRAINED="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"

echo "==============================="
echo " TRELLIS2 — Shape Latent Encoding"
echo " Root:        $ROOT"
echo " Resolution:  $RESOLUTION"
echo " Encoder:     $ENC_PRETRAINED"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# 4.1 — Generate dual_grid_512 (higher-resolution O-Voxels from existing meshes)
#        dual_grid_256 is enough for SC-VAE; flow model needs 512 for better quality.
# --------------------------------------------------------------------------
echo ""
echo "[1/3] Converting meshes to O-Voxels at resolution $RESOLUTION..."
echo "      (This re-uses existing mesh_dumps — no Blender needed here)"
python data_toolkit/dual_grid.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --resolution "$RESOLUTION"

# --------------------------------------------------------------------------
# 4.1b — Update metadata so dual_grid_converted=True is visible to encode step
# --------------------------------------------------------------------------
echo ""
echo "[1b/3] Updating metadata after dual_grid_512..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

# --------------------------------------------------------------------------
# 4.2 — Encode shape latents using pretrained TRELLIS.2-4B encoder
#        Output: shape_latents/shape_enc_next_dc_f16c32_fp16_512/
#        Requires GPU (CUDA) and ~3-5 GB VRAM
# --------------------------------------------------------------------------
echo ""
echo "[2/3] Encoding shape latents (pretrained encoder, GPU)..."
echo "      NOTE: First run will download ~2 GB from HuggingFace"
python data_toolkit/encode_shape_latent.py \
    --root "$ROOT" \
    --resolution "$RESOLUTION" \
    --enc_pretrained "$ENC_PRETRAINED"

# --------------------------------------------------------------------------
# 4.3 — Update metadata to reflect encoded shape latents
# --------------------------------------------------------------------------
echo ""
echo "[3/3] Updating metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

echo ""
echo "==============================="
echo " Shape latent encoding COMPLETE"
echo " Latents at: $ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_${RESOLUTION}/"
echo "==============================="
