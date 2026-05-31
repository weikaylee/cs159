#!/usr/bin/env bash
# Phase 2 EMRDM training from scratch on mirror-space data. Calls train_emrdm.py, which calls main.py.
#
# Prerequisites:
#   1. prepare_mirror_dataset.slurm must have completed successfully so that
#      ${DATA_ROOT}/all_{train,val,test}_paths_mirror.pkl exist.
#
# Monitor:  squeue -u $USER
# Output:   tail -f ${LOG_DIR}/slurm-emrdm-train-<jobid>.out
#
# TODO: update --epochs and global vars to match local paths

# ── Resource allocation ────────────────────────────────────────────────────────
#SBATCH --job-name=phase2-emrdm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=epyc
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-emrdm-train-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-emrdm-train-%j.err

set -euo pipefail

REPO_ROOT="/resnick/groups/perona/oywang/cs159"
DATA_ROOT="${REPO_ROOT}/data/mirror_s2"
CODE_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="${REPO_ROOT}/runs"

mkdir -p "${LOG_DIR}"
# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc 
# NOTE: use this one olivia! (delete the line below) i think; uses ur conda path
# source /home/kwei2/miniforge3/etc/profile.d/conda.sh
conda activate emrdm

# ── Training ──────────────────────────────────────────────────────────────────
cd "${CODE_DIR}"
python train_emrdm.py \
    --data_root  "${DATA_ROOT}" \
    --devices    2 \
    --max_epochs 5 \
    --wandb \
    --logdir     "${LOG_DIR}/emrdm_train"

echo "===== Finished: $(date) ====="
