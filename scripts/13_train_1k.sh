#!/usr/bin/env bash
# Shape Flow training on 1 001-object dataset (ObjaverseXL Sketchfab)
#
# Model:    ElasticSLatFlowModel — 1.3B params (mesma arquitetura do 10k)
# Config:   configs/gen/slat_flow_img2shape_1k.json
# Steps:    30 000 default  (~120 epochs sobre 1k objetos × 8 views / batch 4)
#           Ajustável: MAX_STEPS=50000 bash scripts/13_train_1k.sh
# Hardware: RTX 3090 (24 GB) — single GPU
#
# Para retomar após interrupção, basta re-executar este script;
# o trainer detecta automaticamente o último checkpoint em OUTPUT_DIR.
#
# Run from repo root: bash scripts/13_train_1k.sh
set -e

CONDA_ENV="trellis2"
ROOT="datasets/ObjaverseXL_sketchfab"
SHAPE_LATENT_DIR="$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
RENDER_COND_DIR="$ROOT/renders_cond"
CONFIG="configs/gen/ablations/E_combined_reuse_mixer_mask50.json" # configs/gen/slat_flow_img2shape_1k.json"
OUTPUT_DIR="results/E_combined_reuse_mixer_mask75_1k_big"

MAX_STEPS=${MAX_STEPS:-30000}

echo "========================================"
echo " TRELLIS2 — Shape Flow Training (1k)"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Steps:      $MAX_STEPS"
echo " Model:      1.3B params (full architecture)"
echo " Precision:  bfloat16 (AMP)"
echo " Batch:      2 per GPU × 2 splits = effective batch 4"
echo " Checkpoints + samples every 2 000 steps"
echo "========================================"
echo ""

# Epoch estimate: 1001 objects × 8 views = 8008 images; batch 4 → ~2002 steps/epoch
EPOCHS=$(python -c "print(f'{$MAX_STEPS / 2002:.1f}')" 2>/dev/null || echo "~15")
echo "Estimated epochs : ~$EPOCHS  (~30k steps / 2002 steps-per-epoch)"
echo "Estimated time   : ~1-2 s/step on RTX 3090 → 8-17 h for 30k steps"
echo ""
echo "To monitor: tensorboard --logdir $OUTPUT_DIR/tb_logs"
echo ""

# Verify dataset readiness
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
print(f'  Shape latents : {enc}')
print(f'  Cond renders  : {rnd}')
print(f'  Ready         : {ready}')
if ready < 100:
    print('  ERROR: fewer than 100 objects ready — run scripts/12_build_dataset_1k.sh first')
    sys.exit(1)
"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override max_steps at runtime if different from default
if [ "$MAX_STEPS" != "30000" ]; then
    echo "Overriding max_steps to $MAX_STEPS..."
    python -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['trainer']['args']['max_steps'] = $MAX_STEPS
tmp = '/tmp/slat_flow_1k_override.json'
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
