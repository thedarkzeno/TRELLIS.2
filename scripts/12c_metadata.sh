#!/usr/bin/env bash
# Run Stage 8 only: update metadata + print training-ready stats
# Usage: bash scripts/12c_metadata.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SOURCE="sketchfab"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

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
