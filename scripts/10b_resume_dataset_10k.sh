#!/usr/bin/env bash
# Resume dataset pipeline after Windows reboot interruption
#
# State at interruption (confirmed 2026-03-11):
#   Stage 1 (metadata init)  — DONE
#   Stage 2 (download)       — DONE  (19 236 objects in raw/)
#   Stage 3 (mesh dump)      — DONE  (19 236 pickles in mesh_dumps/)
#   Stage 4 (PBR dump)       — INTERRUPTED at ~4 988 / 19 236 objects
#   Stages 5-8               — NOT STARTED
#
# This script resumes from Stage 4 (PBR dump is idempotent — skips done objects)
# and runs all remaining stages through to completion.
#
# Run from repo root:  bash scripts/10b_resume_dataset_10k.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

# Parallel workers for Blender-based stages (mesh, PBR, render)
MAX_WORKERS=8

echo "========================================"
echo " TRELLIS2 — Dataset Resume (Stage 4→8)"
echo " Root:     $ROOT"
echo " Workers:  $MAX_WORKERS (Blender stages)"
echo " Resuming: PBR dump (~14 248 objects remaining)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# STAGE 4 — Dump PBR textures via Blender  (RESUMING)
#   Already done: ~4 988 / 19 236 objects
#   Remaining:    ~14 248 objects
#   Estimated time remaining: ~14 248 / 8 workers × 30 s/object ≈ 15 h
# --------------------------------------------------------------------------
echo ""
echo "[4/8] Resuming PBR texture dump (Blender, $MAX_WORKERS workers)..."
echo "      Skipping ~4 988 already-processed objects (idempotent)."
python data_toolkit/dump_pbr.py ObjaverseXL \
    --root "$ROOT" \
    --source "$SOURCE" \
    --max_workers "$MAX_WORKERS"

# Update metadata after Blender stages
python data_toolkit/build_metadata.py ObjaverseXL --source "$SOURCE" --root "$ROOT"

# --------------------------------------------------------------------------
# STAGE 5 — Convert meshes → O-Voxels at resolution 512
#   CPU-only (< 10 s/object) — 19 236 objects ≈ 2-3 h
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
#   ~19 236 objects × 0.1 s ≈ 30-40 min on RTX 3090
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
#   ~19 236 objects × 60 s / 8 workers ≈ 40 h
#   TIP: parallelise this step manually — see FULLRUN.md §Parallelização
# --------------------------------------------------------------------------
echo ""
echo "[7/8] Rendering multi-view conditions (Blender, $MAX_WORKERS workers)..."
echo "      NOTE: This is the slowest step (~20-40 h for ~19k objects)"
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
