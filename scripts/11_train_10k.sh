#!/usr/bin/env bash
# Full Shape Flow training on ~10k object dataset
#
# Model:   ElasticSLatFlowModel — 1.3B params (full production architecture)
# Config:  configs/gen/slat_flow_img2shape_10k.json
# Steps:   100 000 (adjustable via MAX_STEPS below)
# Hardware: RTX 3090 (24 GB) — single GPU
#
# Run from repo root:  bash scripts/11_train_10k.sh
# To resume after interruption, just re-run this script; it automatically
# picks up the latest checkpoint in OUTPUT_DIR.
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SHAPE_LATENT_DIR="$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
RENDER_COND_DIR="$ROOT/renders_cond"
CONFIG="configs/gen/slat_flow_img2shape_10k.json"
OUTPUT_DIR="results/slat_flow_shape_10k"

# Override max_steps at runtime if needed:
#   MAX_STEPS=50000 bash scripts/11_train_10k.sh
MAX_STEPS=${MAX_STEPS:-100000}

echo "========================================"
echo " TRELLIS2 — Shape Flow Training (10k)"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Steps:      $MAX_STEPS"
echo " Model:      1.3B params (full architecture)"
echo " Precision:  bfloat16 (AMP)"
echo " Batch:      2 per GPU × 2 splits = effective batch 4"
echo "========================================"
echo ""
echo "Estimated training time:"
echo "  ~1-2 s/step on RTX 3090 → 28-56 h for 100k steps"
echo "  Checkpoints + samples every 5 000 steps"
echo ""
echo "To monitor: tensorboard --logdir $OUTPUT_DIR/tb_logs"
echo ""

# Verify data is ready
echo "Checking dataset readiness..."
python -c "
import pandas as pd, os, glob, sys

root = '$ROOT'

sl_csvs = glob.glob(os.path.join(root, 'shape_latents', '*', 'metadata.csv'))
enc = 0
if sl_csvs:
    sl_df = pd.concat([pd.read_csv(f) for f in sl_csvs]).drop_duplicates('sha256')
    enc = int((sl_df.get('shape_latent_encoded', pd.Series(dtype=bool)) == True).sum())

rc_path = os.path.join(root, 'renders_cond', 'metadata.csv')
rnd = 0
if os.path.exists(rc_path):
    rc_df = pd.read_csv(rc_path)
    rnd = int((rc_df.get('cond_rendered', pd.Series(dtype=bool)) == True).sum())

ready = min(enc, rnd)
print(f'  Objects ready: {ready}')
if ready < 100:
    print('  WARNING: fewer than 100 objects ready — run 10_build_dataset_10k.sh first')
    sys.exit(1)
"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Patch max_steps if overridden at runtime
if [ "$MAX_STEPS" != "100000" ]; then
    echo "Overriding max_steps to $MAX_STEPS..."
    python -c "
import json, sys
cfg = json.load(open('$CONFIG'))
cfg['trainer']['args']['max_steps'] = $MAX_STEPS
tmp = '/tmp/slat_flow_10k_override.json'
json.dump(cfg, open(tmp, 'w'), indent=4)
print(tmp)
" > /tmp/config_path.txt
    CONFIG_USED=$(cat /tmp/config_path.txt)
else
    CONFIG_USED="$CONFIG"
fi

python train.py \
    --config "$CONFIG_USED" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"$ROOT\", \"shape_latent\": \"$SHAPE_LATENT_DIR\", \"render_cond\": \"$RENDER_COND_DIR\"}}"

echo ""
echo "========================================"
echo " Training COMPLETE"
echo " Outputs: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Latest checkpoints:"
ls -lht "$OUTPUT_DIR/ckpts/" 2>/dev/null | head -8 || echo "  (none yet)"
