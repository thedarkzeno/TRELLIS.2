#!/usr/bin/env bash
# Full dataset pipeline for ~10 000 objects (all stages)
#
# Idempotent: each stage skips already-processed objects.
# Run from repo root:  bash scripts/10_build_dataset_10k.sh
#
# To parallelise heavy stages (mesh/PBR/render) across multiple terminals,
# set RANK and WORLD_SIZE_PARALLEL manually — see FULLRUN.md for details.
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

# world_size for download: 168 307 total / 17 ≈ 9 900 objects
DOWNLOAD_WORLD_SIZE=17

# Parallel workers for Blender-based stages (mesh, PBR, render)
# Increase if you have more CPU cores. Blender is single-threaded per call,
# so more workers = more simultaneous Blender processes.
MAX_WORKERS=8

echo "========================================"
echo " TRELLIS2 — Full Dataset Build (10k)"
echo " Root:        $ROOT"
echo " Objects:     ~9 900 (world_size=$DOWNLOAD_WORLD_SIZE)"
echo " Workers:     $MAX_WORKERS (Blender stages)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# STAGE 1 — Initialize/update metadata
# --------------------------------------------------------------------------
echo ""
echo "[1/8] Initializing metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 2 — Download ~10k objects (rank=0 processes 1/DOWNLOAD_WORLD_SIZE)
# --------------------------------------------------------------------------
echo ""
echo "[2/8] Downloading ~9 900 objects (this may take several hours)..."
python data_toolkit/download.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --world_size "$DOWNLOAD_WORLD_SIZE"

# Update metadata after download
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 3 — Dump meshes via Blender
#   Estimated time: ~10 000 objects / 8 workers × 30 s/object ≈ 10 h
# --------------------------------------------------------------------------
echo ""
echo "[3/8] Dumping meshes (Blender, $MAX_WORKERS workers)..."
python data_toolkit/dump_mesh.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --max_workers "$MAX_WORKERS"

# --------------------------------------------------------------------------
# STAGE 4 — Dump PBR textures via Blender
#   Estimated time: similar to mesh dump, ~10 h
# --------------------------------------------------------------------------
echo ""
echo "[4/8] Dumping PBR textures (Blender, $MAX_WORKERS workers)..."
python data_toolkit/dump_pbr.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --max_workers "$MAX_WORKERS"

# Update metadata after Blender stages
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 5 — Convert meshes → O-Voxels at resolution 512
#   CPU-only, fast (< 10 s/object) — no GPU needed
# --------------------------------------------------------------------------
echo ""
echo "[5/8] Converting meshes to O-Voxels at resolution 512..."
python data_toolkit/dual_grid.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --resolution 512

python data_toolkit/asset_stats.py --root "$ROOT"

# Update metadata
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 6 — Encode shape latents with pretrained TRELLIS.2-4B encoder (GPU)
#   Estimated time: ~10 000 objects × 0.1 s/object ≈ 15–20 min on RTX 3090
#   NOTE: requires HuggingFace login (huggingface-cli login)
# --------------------------------------------------------------------------
echo ""
echo "[6/8] Encoding shape latents (GPU, pretrained TRELLIS.2-4B encoder)..."
python data_toolkit/encode_shape_latent.py \
    --root "$ROOT" \
    --resolution 512 \
    --enc_pretrained "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"

# Update metadata
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 7 — Render multi-view image conditions (Blender)
#   Estimated time: ~10 000 objects × 60 s × (1/8 workers) ≈ 20 h
#   TIP: parallelise this step manually — see FULLRUN.md §Parallelização
# --------------------------------------------------------------------------
echo ""
echo "[7/8] Rendering multi-view conditions (Blender, $MAX_WORKERS workers)..."
echo "      NOTE: This is the slowest step (~10-20 h for 10k objects)"
python data_toolkit/render_cond.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --num_cond_views 16 \
    --max_workers "$MAX_WORKERS"

# --------------------------------------------------------------------------
# STAGE 8 — Final metadata update
# --------------------------------------------------------------------------
echo ""
echo "[8/8] Final metadata update..."
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

echo ""
echo "========================================"
echo " Dataset build COMPLETE"
echo " Objects ready for training:"
python -c "
import pandas as pd
df = pd.read_csv('$ROOT/metadata.csv')
enc = (df['shape_latent_encoded'] == True).sum()
rnd = (df['cond_rendered'] == True).sum()
both = ((df['shape_latent_encoded'] == True) & (df['cond_rendered'] == True)).sum()
print(f'  Shape latents encoded : {enc}')
print(f'  Cond renders done     : {rnd}')
print(f'  Ready for flow training: {both}')
"
echo "========================================"
