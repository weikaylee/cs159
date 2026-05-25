#!/bin/bash
# Phase 2 EMRDM prediction: best VGG and best Spectral config only (single jobs, not array)
# This script runs one VGG-style and one Spectral-style config with optimal settings.
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, and the paths below.

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J "phase2-emrdm-predict"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-emrdm-predict-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-emrdm-predict-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase2_emrdm"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/emrdm_predict"
mkdir -p "$OUTPUT_ROOT"

cd "$CODE_DIR"
python run_emrdm_predict.py \
        --data_root /resnick/groups/perona/oywang/cs159/data \
        --ckpt_path /resnick/groups/perona/oywang/cs159/emrdm_weights/train/sentinel/checkpoints \
        --output_dir /resnick/groups/perona/oywang/cs159/output/emrdm_predict
EMRDM_EXIT=$?
if [ $EMRDM_EXIT -ne 0 ]; then
    echo "ERROR: EMRDM prediction job failed with exit code $EMRDM_EXIT" >&2
    exit $EMRDM_EXIT
fi
