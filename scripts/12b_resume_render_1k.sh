#!/usr/bin/env bash
# Resume dataset pipeline from Stage 7 (render multi-view conditions)
#
# State at interruption (confirmed 2026-03-11):
#   Stages 1-6  — DONE  (1 003 objects: mesh, PBR, dual_grid_512, shape_latents)
#   Stage 7     — ~70 / 1 001 renders done
#   Stage 8     — NOT STARTED
#
# GPU rendering investigation summary (WSL2):
#   - Cycles CUDA: get_devices() returns empty in WSL2 headless (known limitation)
#   - EEVEE + DISPLAY=:0: uses Mesa LLVMpipe (software), NOT NVIDIA GPU
#   - WSL2 does not install NVIDIA libGL — only libcuda is available
#   Result: CPU rendering is the only working option in WSL2.
#
# CPU optimizations applied:
#   - Resolution: 512 (vs 1024) — 4x fewer pixels, same quality for conditioning
#   - Views: 8 (vs 16) — sufficient for multi-view conditioning
#   - Workers: 12 — saturate all 12 threads of the Ryzen 5600X
#   - Estimated time: ~931 × (8 views × ~8s/CPU) / 12 workers ≈ ~1.6 h
#
# Run from repo root:  bash scripts/12b_resume_render_1k.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

CYCLES_DEVICE="CPU"
MAX_WORKERS=12      # Ryzen 5600X: 6 cores / 12 threads
COND_RESOLUTION=512 # reduced from 1024 — still sufficient for conditioning
NUM_VIEWS=8         # reduced from 16 — sufficient for multi-view conditioning

echo "========================================"
echo " TRELLIS2 — Resume Renders (1k, CPU)"
echo " Root:          $ROOT"
echo " Engine:        Cycles CPU"
echo " Workers:       $MAX_WORKERS (Ryzen 5600X 12T)"
echo " Resolution:    ${COND_RESOLUTION}x${COND_RESOLUTION}"
echo " Views/object:  $NUM_VIEWS"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Progress check
DONE=$(find "$ROOT/renders_cond" -name "transforms.json" 2>/dev/null | wc -l)
TOTAL=1001
REMAINING=$((TOTAL - DONE))
echo ""
echo "Renders so far: $DONE / $TOTAL  ($REMAINING remaining)"

# --------------------------------------------------------------------------
# STAGE 7 — Render multi-view image conditions (Blender Cycles, CPU)
#   Resolution 512 and 8 views are standard for conditioning datasets.
#   Idempotent: already-rendered objects are skipped automatically.
# --------------------------------------------------------------------------
echo ""
echo "[7/8] Rendering multi-view conditions (Cycles CPU, $MAX_WORKERS workers)..."
python data_toolkit/render_cond.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --num_cond_views "$NUM_VIEWS" \
    --cond_resolution "$COND_RESOLUTION" \
    --max_workers "$MAX_WORKERS" \
    --cycles-device "$CYCLES_DEVICE"

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
import pandas as pd, os, glob

root = '$ROOT'

# shape_latent_encoded lives in shape_latents/<model>/metadata.csv
sl_pattern = os.path.join(root, 'shape_latents', '*', 'metadata.csv')
sl_csvs = glob.glob(sl_pattern)
if sl_csvs:
    sl_df = pd.concat([pd.read_csv(f) for f in sl_csvs]).drop_duplicates('sha256').set_index('sha256')
    enc = int((sl_df.get('shape_latent_encoded', pd.Series(dtype=bool)) == True).sum())
else:
    enc = 0

# cond_rendered lives in renders_cond/metadata.csv
rc_path = os.path.join(root, 'renders_cond', 'metadata.csv')
if os.path.exists(rc_path):
    rc_df = pd.read_csv(rc_path).set_index('sha256')
    rnd = int((rc_df.get('cond_rendered', pd.Series(dtype=bool)) == True).sum())
else:
    rnd = 0

# intersection
if sl_csvs and os.path.exists(rc_path):
    both = len(sl_df[sl_df.get('shape_latent_encoded', False) == True].index.intersection(
               rc_df[rc_df.get('cond_rendered', False) == True].index))
else:
    both = 0

print(f'  Shape latents encoded  : {enc}')
print(f'  Cond renders done      : {rnd}')
print(f'  Ready for flow training: {both}')
"
echo "========================================"
