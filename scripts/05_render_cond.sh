#!/usr/bin/env bash
# Stage 5 — Render multi-view image conditions via Blender
# Prerequisites: 01_build_dataset.sh must have been run (raw files + mesh_dumps exist)
# Run from repo root: bash scripts/05_render_cond.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"
NUM_COND_VIEWS=16   # number of viewpoints to render per object (default: 16)

echo "==============================="
echo " TRELLIS2 — Render Conditions"
echo " Root:       $ROOT"
echo " Views/obj:  $NUM_COND_VIEWS"
echo "==============================="
echo ""
echo "NOTE: Blender 3.0.1 will be downloaded to /tmp if not already present (~200 MB)."
echo "      Rendering 21 objects × 16 views takes ~10-30 min on CPU."
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# 5.1 — Render conditional views for each object
#        Output: renders_cond/<sha256>/  (PNG images + transforms.json per object)
#        Uses Blender headless; downloads Blender 3.0.1 to /tmp if needed.
# --------------------------------------------------------------------------
echo "[1/2] Rendering multi-view conditions..."
python data_toolkit/render_cond.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --num_cond_views "$NUM_COND_VIEWS"

# --------------------------------------------------------------------------
# 5.2 — Update metadata to reflect rendered conditions
# --------------------------------------------------------------------------
echo ""
echo "[2/2] Updating metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

echo ""
echo "==============================="
echo " Render conditions COMPLETE"
echo " Images at: $ROOT/renders_cond/"
echo "==============================="
