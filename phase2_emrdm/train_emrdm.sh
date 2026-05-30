#!/usr/bin/env bash
# Phase 2 EMRDM training from scratch on mirror-space data.
#
# Prerequisites:
#   1. prepare_mirror_dataset.slurm must have completed successfully so that
#      ${DATA_ROOT}/all_{train,val,test}_paths_mirror.pkl exist.
#   2. Edit the USER SETTINGS block below, then:
#        sbatch phase2_emrdm/train_emrdm.sh
#
# Monitor:  squeue -u $USER
# Output:   tail -f ${LOG_DIR}/slurm-emrdm-train-<jobid>.out

# ── Resource allocation ────────────────────────────────────────────────────────
#SBATCH --job-name=phase2-emrdm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mail-user=kayleewei023@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/slurm-emrdm-train-%j.out
#SBATCH --error=logs/slurm-emrdm-train-%j.err

set -euo pipefail

# Resolve paths relative to repo root (one level above phase2_emrdm/).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
CODE_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"
# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# Redirect large caches away from home.
export HF_HOME=/resnick/groups/perona/${USER}/.cache/huggingface
export TORCH_HOME=/resnick/groups/perona/${USER}/.cache/torch
export PIP_CACHE_DIR=/resnick/groups/perona/${USER}/.cache/pip

# ── Job info ──────────────────────────────────────────────────────────────────
echo "===== Job info ====="
echo "  Node:      $(hostname)"
echo "  GPUs:      ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Data root: ${DATA_ROOT}"
echo "  Code dir:  ${CODE_DIR}"
echo "  Log dir:   ${LOG_DIR}"
echo "  Start:     $(date)"
echo "===================="

# ── Training ──────────────────────────────────────────────────────────────────
cd "${CODE_DIR}"
python train_emrdm.py \
    --data_root "${DATA_ROOT}" \
    --devices   2 \
    --wandb \
    --logdir    "${LOG_DIR}/emrdm_train"

echo "===== Finished: $(date) ====="
