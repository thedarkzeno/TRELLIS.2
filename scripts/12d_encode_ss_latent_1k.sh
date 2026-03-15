#!/usr/bin/env bash
# Encode Sparse Structure (SS) latents for ~1 000 objects
#
# This script fills the gap in 12_build_dataset_1k.sh: the shape latent
# pipeline (script 12, stage 6) produces per-voxel features used by the
# SLat flow model, but the SS flow model needs a separate, lower-resolution
# latent derived from the binary voxel occupancy.
#
# Pipeline:
#   shape_latents/{name}/{sha256}.npz  (coords only, from TRELLIS.2-4B encoder)
#       |
#       v  encode_ss_latent.py  — pretrained TRELLIS-image-large SS encoder (GPU)
#       |
#   ss_latents/{enc_name}_{res}/{sha256}.npz  (z: [8, 16, 16, 16])
#
# Prerequisite: run scripts/12_build_dataset_1k.sh first so that
#   - shape_latents exist (shape_latent_encoded == True in metadata)
#   - cond renders exist  (cond_rendered    == True in metadata)
#
# Idempotent: encode_ss_latent.py skips already-processed objects.
# Run from repo root:  bash scripts/12d_encode_ss_latent_1k.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

# Must match the shape latent directory produced in 12_build_dataset_1k.sh stage 6
SHAPE_LATENT_NAME="shape_enc_next_dc_f16c32_fp16_512"

# SS encoder from TRELLIS v1 (publicly available, no extra login needed beyond HF)
SS_ENC_PRETRAINED="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"

# Resolution fed into the SS encoder (matches pretrained model's expected input)
SS_RESOLUTION=64

echo "========================================"
echo " TRELLIS2 — SS Latent Encoding (1k)"
echo " Root:          $ROOT"
echo " Shape latents: $SHAPE_LATENT_NAME"
echo " SS encoder:    $SS_ENC_PRETRAINED"
echo " SS resolution: $SS_RESOLUTION"
echo " Estimated time: ~2-5 min on RTX 3090 for 1k objects"
echo "========================================"
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# STAGE 1 — Verify prerequisites: shape latents must exist
# --------------------------------------------------------------------------
echo "[1/3] Checking prerequisites..."
python -c "
import pandas as pd, os, glob, sys

root = '$ROOT'
sl_csvs = glob.glob(os.path.join(root, 'shape_latents', '*', 'metadata.csv'))
enc = 0
if sl_csvs:
    sl_df = pd.concat([pd.read_csv(f) for f in sl_csvs]).drop_duplicates('sha256')
    enc = int((sl_df.get('shape_latent_encoded', pd.Series(dtype=bool)) == True).sum())

print(f'  Shape latents encoded : {enc}')
if enc < 10:
    print('  ERROR: fewer than 10 shape latents found.')
    print('  Run scripts/12_build_dataset_1k.sh first (stages 1-6).')
    sys.exit(1)
print('  OK — shape latents found, proceeding.')
"

# --------------------------------------------------------------------------
# STAGE 2 — Encode SS latents  (~2-5 min for 1k objects on RTX 3090)
# --------------------------------------------------------------------------
echo ""
echo "[2/3] Encoding SS latents (GPU)..."
python data_toolkit/encode_ss_latent.py \
    --root "$ROOT" \
    --shape_latent_root "$ROOT" \
    --shape_latent_name "$SHAPE_LATENT_NAME" \
    --resolution "$SS_RESOLUTION" \
    --enc_pretrained "$SS_ENC_PRETRAINED"

# --------------------------------------------------------------------------
# STAGE 3 — Update metadata so ss_latent_encoded appears in CSVs
# --------------------------------------------------------------------------
echo ""
echo "[3/3] Updating metadata..."
python data_toolkit/build_metadata.py ObjaverseXL \
    --source "$SOURCE" \
    --root "$ROOT"

echo ""
echo "========================================"
echo " SS Latent Encoding COMPLETE"
python -c "
import pandas as pd, os, glob

root = '$ROOT'

ss_csvs = glob.glob(os.path.join(root, 'ss_latents', '*', 'metadata.csv'))
ss_enc = 0
if ss_csvs:
    ss_df = pd.concat([pd.read_csv(f) for f in ss_csvs]).drop_duplicates('sha256')
    ss_enc = int((ss_df.get('ss_latent_encoded', pd.Series(dtype=bool)) == True).sum())

rc_path = os.path.join(root, 'renders_cond', 'metadata.csv')
rnd = 0
if os.path.exists(rc_path):
    rc_df = pd.read_csv(rc_path)
    rnd = int((rc_df.get('cond_rendered', pd.Series(dtype=bool)) == True).sum())

ready = min(ss_enc, rnd)
print(f'  SS latents encoded : {ss_enc}')
print(f'  Cond renders done  : {rnd}')
print(f'  Ready for SS flow  : {ready}')
if ready > 0:
    print('')
    print('  Next step: bash scripts/15_train_ss_flow_1k.sh')
"
echo "========================================"
