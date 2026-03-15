#!/usr/bin/env bash
# Sparse Structure Flow training on 1 001-object dataset (ObjaverseXL Sketchfab)
#
# Model:    ElasticSparseStructureFlowModel — 4 unique blocks, reuse [1,3,3,1]
#           (8 effective depth) + patch mixer (2 blocks, dim=768) + mask_ratio=0.5
# Config:   configs/gen/ss_flow_optimized_1k.json
# Steps:    30 000 default
#           Ajustável: MAX_STEPS=50000 bash scripts/15_train_ss_flow_1k.sh
# Hardware: RTX 3090 (24 GB) — single GPU
#
# Prerequisites (run once before training):
#   1. bash scripts/12_build_dataset_1k.sh   (shape latents + cond renders)
#   2. bash scripts/12d_encode_ss_latent_1k.sh  (SS latents)
#
# Para retomar após interrupção, basta re-executar este script;
# o trainer detecta automaticamente o último checkpoint em OUTPUT_DIR.
#
# Run from repo root: bash scripts/15_train_ss_flow_1k.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"

# SS latent dir produced by scripts/12d_encode_ss_latent_1k.sh
# Format: ss_latents/{enc_name}_{resolution}
SS_LATENT_DIR="$ROOT/ss_latents/ss_enc_conv3d_16l8_fp16_64"

RENDER_COND_DIR="$ROOT/renders_cond"
CONFIG="configs/gen/ss_flow_optimized_1k.json"
OUTPUT_DIR="results/ss_flow_optimized_1k"

MAX_STEPS=${MAX_STEPS:-30000}

echo "========================================"
echo " TRELLIS2 — SS Flow Training (1k)"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Steps:      $MAX_STEPS"
echo " Model:      ElasticSparseStructureFlowModel"
echo "             4 blocks reuse [1,3,3,1] = 8 effective depth"
echo "             Patch mixer: 2 blocks (dim=768)"
echo " Mask ratio: 0.5 (50% token drop during training)"
echo " Precision:  bfloat16 (AMP)"
echo " Batch:      8 per GPU / 2 splits = effective batch 4"
echo " Checkpoints + samples every 1 000 steps"
echo "========================================"
echo ""

# Epoch estimate: 1001 objects × 1 view per step / batch 4 ≈ 250 steps/epoch
EPOCHS=$(python -c "print(f'{$MAX_STEPS / 250:.1f}')" 2>/dev/null || echo "~120")
echo "Estimated epochs : ~$EPOCHS  (~30k steps / 250 steps-per-epoch)"
echo "Estimated time   : ~0.3-0.8 s/step on RTX 3090 → 2-7 h for 30k steps"
echo ""
echo "To monitor: tensorboard --logdir $OUTPUT_DIR/tb_logs"
echo ""

# --------------------------------------------------------------------------
# Verify dataset readiness
# --------------------------------------------------------------------------
echo "Checking dataset readiness..."
python -c "
import pandas as pd, os, glob, sys

root = '$ROOT'

# Check SS latents
ss_csvs = glob.glob(os.path.join(root, 'ss_latents', '*', 'metadata.csv'))
ss_enc = 0
if ss_csvs:
    ss_df = pd.concat([pd.read_csv(f) for f in ss_csvs]).drop_duplicates('sha256')
    ss_enc = int((ss_df.get('ss_latent_encoded', pd.Series(dtype=bool)) == True).sum())

# Check cond renders
rc_path = os.path.join(root, 'renders_cond', 'metadata.csv')
rnd = 0
if os.path.exists(rc_path):
    rc_df = pd.read_csv(rc_path)
    rnd = int((rc_df.get('cond_rendered', pd.Series(dtype=bool)) == True).sum())

ready = min(ss_enc, rnd)
print(f'  SS latents   : {ss_enc}')
print(f'  Cond renders : {rnd}')
print(f'  Ready        : {ready}')
if ready < 100:
    print('')
    print('  ERROR: fewer than 100 objects ready.')
    if ss_enc < 100:
        print('  → Run scripts/12d_encode_ss_latent_1k.sh to encode SS latents.')
    if rnd < 100:
        print('  → Run scripts/12_build_dataset_1k.sh first (includes stage 7 renders).')
    sys.exit(1)
print('  OK — dataset is ready.')
"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------------------------------
# Override max_steps at runtime if different from default
# --------------------------------------------------------------------------
if [ "$MAX_STEPS" != "30000" ]; then
    echo "Overriding max_steps to $MAX_STEPS..."
    python -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['trainer']['args']['max_steps'] = $MAX_STEPS
tmp = '/tmp/ss_flow_1k_override.json'
json.dump(cfg, open(tmp, 'w'), indent=4)
print(tmp)
" > /tmp/config_path.txt
    CONFIG_USED=$(cat /tmp/config_path.txt)
else
    CONFIG_USED="$CONFIG"
fi

# --------------------------------------------------------------------------
# Launch training
# --------------------------------------------------------------------------
python train.py \
    --config "$CONFIG_USED" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"$ROOT\", \"ss_latent\": \"$SS_LATENT_DIR\", \"render_cond\": \"$RENDER_COND_DIR\"}}"

echo ""
echo "========================================"
echo " Training COMPLETE"
echo " Outputs: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Latest checkpoints:"
ls -lht "$OUTPUT_DIR/ckpts/" 2>/dev/null | head -8 || echo "  (none yet)"
