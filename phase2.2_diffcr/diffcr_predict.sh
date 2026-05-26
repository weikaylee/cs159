#!/bin/bash
# Phase 2 EMRDM prediction: best VGG and best Spectral config only (single jobs, not array)
# This script runs one VGG-style and one Spectral-style config with optimal settings.
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, and the paths below.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J "phase2-diffcr-predict"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-diffcr-predict-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-diffcr-predict-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate uncrtaints

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/diffcr-models/UnCRtainTS/"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/diffcr_predict"
mkdir -p "$OUTPUT_ROOT"

cd "$CODE_DIR"
python model/test_reconstruct.py \
  --experiment_name diffcr_bs32_epoch17 \
  --root1 "$DATA_ROOT" \
  --root2 "$DATA_ROOT" \
  --root3 "$DATA_ROOT" \
  --precomputed "$CODE_DIR/util/precomputed" \
  --input_t 1 \
  --region all \
  --export_every 1 \
  --res_dir ./inference \
  --weight_folder checkpoint/ \
  --pretrain \
  --sample_type pretrain \
  --device cuda:0 \
  --use_sar \
  --out_conv 13
DIFFCR_EXIT=$?
if [ $DIFFCR_EXIT -ne 0 ]; then
    echo "ERROR: DIFFCR prediction job failed with exit code $DIFFCR_EXIT" >&2
    exit $DIFFCR_EXIT
fi
