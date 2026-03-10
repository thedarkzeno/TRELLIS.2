#!/usr/bin/env bash
# Stage 1 — Dataset preparation for validation (20 objects)
# Run from repo root: bash scripts/01_build_dataset.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

# Number of objects ≈ total_in_csv / WORLD_SIZE
# TRELLIS-500K sketchfab CSV has 168307 rows → world_size=8000 gives ~21 objects
WORLD_SIZE=8000

echo "==============================="
echo " TRELLIS2 — Dataset Preparation"
echo " Root:       $ROOT"
echo " Source:     $SOURCE"
echo " World size: $WORLD_SIZE (~20 objects)"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# The data_toolkit scripts need to find the `datasets` package
# Python adds data_toolkit/ to sys.path when running scripts from that dir
# We run from repo root using `python data_toolkit/<script>.py`

# --------------------------------------------------------------------------
# 2.1 — Initialize metadata (downloads parquet from HuggingFace)
# --------------------------------------------------------------------------
echo ""
echo "[1/6] Building initial metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

# --------------------------------------------------------------------------
# 2.2 — Download ~20 objects
# --------------------------------------------------------------------------
echo ""
echo "[2/6] Downloading objects (world_size=$WORLD_SIZE, rank=0)..."
python data_toolkit/download.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --world_size "$WORLD_SIZE"

# --------------------------------------------------------------------------
# 2.3 — Update metadata after download
# --------------------------------------------------------------------------
echo ""
echo "[3/6] Updating metadata after download..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

# --------------------------------------------------------------------------
# 2.4 — Dump meshes and PBR textures via Blender
# --------------------------------------------------------------------------
echo ""
echo "[4/6] Dumping meshes (Blender)..."
python data_toolkit/dump_mesh.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE"

echo ""
echo "[4b/6] Dumping PBR textures (Blender)..."
python data_toolkit/dump_pbr.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE"

# --------------------------------------------------------------------------
# 2.5 — Update metadata after mesh/pbr dump
# --------------------------------------------------------------------------
echo ""
echo "[5/6] Updating metadata after mesh/PBR dump..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

# --------------------------------------------------------------------------
# 2.6 — Convert to O-Voxels (resolution 256) + asset stats
# --------------------------------------------------------------------------
echo ""
echo "[6/6] Converting to O-Voxels (resolution=256)..."
python data_toolkit/dual_grid.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --resolution 256

echo ""
echo "[6b/6] Computing asset statistics..."
python data_toolkit/asset_stats.py \
    --root "$ROOT"

# --------------------------------------------------------------------------
# Final metadata update
# --------------------------------------------------------------------------
echo ""
echo "Finalizing metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

echo ""
echo "==============================="
echo " Dataset preparation COMPLETE"
echo " Path: $ROOT"
echo "==============================="
