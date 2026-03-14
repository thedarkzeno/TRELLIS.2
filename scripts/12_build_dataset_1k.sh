#!/usr/bin/env bash
# Full dataset pipeline for ~1 000 objects (all stages)
#
# Idempotent: each stage skips already-processed objects.
# Run from repo root:  bash scripts/12_build_dataset_1k.sh
#
# Disk estimate: ~50-80 GB for 1k objects (raw + meshes + PBR + voxels + renders)
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

# world_size for download: 168 307 total / 168 ≈ 1 002 objects  (rank=0)
DOWNLOAD_WORLD_SIZE=168

# Parallel workers for Blender-based stages (mesh, PBR, render)
MAX_WORKERS=8

echo "========================================"
echo " TRELLIS2 — Full Dataset Build (1k)"
echo " Root:        $ROOT"
echo " Objects:     ~1 000 (world_size=$DOWNLOAD_WORLD_SIZE)"
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
# STAGE 2 — Download ~1k objects (rank=0 processes 1/DOWNLOAD_WORLD_SIZE)
#   Estimated time: ~1 h depending on connection speed
# --------------------------------------------------------------------------
echo ""
echo "[2/8] Downloading ~1 000 objects..."
python data_toolkit/download.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --world_size "$DOWNLOAD_WORLD_SIZE"

# Update metadata after download
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 3 — Dump meshes via Blender
#   Estimated time: ~1 000 objects / 8 workers × 30 s/object ≈ 1 h
# --------------------------------------------------------------------------
echo ""
echo "[3/8] Dumping meshes (Blender, $MAX_WORKERS workers)..."
python data_toolkit/dump_mesh.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --max_workers "$MAX_WORKERS"

# --------------------------------------------------------------------------
# STAGE 4 — Dump PBR textures via Blender
#   Estimated time: ~1 000 objects / 8 workers × 30 s/object ≈ 1 h
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
#   CPU-only (< 10 s/object) — ~1 000 objects ≈ 15 min
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
#   ~1 000 objects × 0.1 s ≈ 2-5 min on RTX 3090
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
#   ~1 000 objects × 60 s / 8 workers ≈ 2 h
# --------------------------------------------------------------------------
echo ""
echo "[7/8] Rendering multi-view conditions (Blender, $MAX_WORKERS workers)..."
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
print(f'  Shape latents encoded  : {enc}')
print(f'  Cond renders done      : {rnd}')
print(f'  Ready for flow training: {both}')
"
echo "========================================"
