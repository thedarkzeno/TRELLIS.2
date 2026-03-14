#!/usr/bin/env bash
# Stage 14 — Real-world inference: image(s) → 3D mesh
#
# Runs the trained SLat flow model on real images using the pretrained
# sparse-structure sampler from TRELLIS.2-4B.
#
# Usage:
#   bash scripts/14_infer.sh [image_glob_or_path ...]
#
# If no image paths are given it falls back to the DEFAULT_IMAGES list below.
#
# Run from repo root: bash scripts/14_infer.sh assets/chair.png
set -e

CONDA_ENV="trellis2"

# ---------------------------------------------------------------------------
# Configuration — edit these to match your experiment
# ---------------------------------------------------------------------------
RESULT_DIR="results/E_combined_reuse_mixer_mask50_1k"   # trained result folder
PRETRAINED="microsoft/TRELLIS.2-4B"                     # or a local cache path
OUTPUT_DIR="${RESULT_DIR}/infer"

STEPS=25
GUIDANCE=3.0
SEED=42
CKPT_STEP=""          # leave empty to use the latest checkpoint automatically

# Default test images (used when no argument is passed)
# Pega as 3 primeiras imagens de exemplo do repositório
mapfile -t DEFAULT_IMAGES < <(ls assets/example_image/*.webp 2>/dev/null | head -3)
if [ ${#DEFAULT_IMAGES[@]} -eq 0 ]; then
    echo "ERRO: nenhuma imagem encontrada em assets/example_image/"
    echo "Passe imagens explicitamente: bash scripts/14_infer.sh foto.jpg"
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Resolve images
# ---------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    IMAGES=("$@")
else
    IMAGES=("${DEFAULT_IMAGES[@]}")
fi

# Build the --images argument list (space-separated for argparse)
IMAGES_ARGS=()
for img in "${IMAGES[@]}"; do
    IMAGES_ARGS+=("$img")
done

# Optional: pass --ckpt_step if set
CKPT_ARGS=()
if [ -n "$CKPT_STEP" ]; then
    CKPT_ARGS=(--ckpt_step "$CKPT_STEP")
fi

echo "========================================"
echo " TRELLIS2 — Real-World Inference"
echo " Result dir: $RESULT_DIR"
echo " Pretrained: $PRETRAINED"
echo " Output dir: $OUTPUT_DIR"
echo " Steps:      $STEPS   Guidance: $GUIDANCE"
echo " Images:     ${IMAGES_ARGS[*]}"
echo "========================================"

python infer.py \
    --images "${IMAGES_ARGS[@]}" \
    --result_dir "$RESULT_DIR" \
    "${CKPT_ARGS[@]}" \
    --pretrained "$PRETRAINED" \
    --output_dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --guidance "$GUIDANCE" \
    --seed "$SEED"

echo ""
echo "Results saved to: $OUTPUT_DIR"
