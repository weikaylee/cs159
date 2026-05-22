#!/usr/bin/env bash
# SLURM job script for EMRDM inference on SEN12MS-CR.
#
# Edit the USER SETTINGS block before submitting, then:
#   sbatch eval_emrdm.sh
#
# Monitor:  squeue -u $USER
# Output:   tail -f logs/eval_emrdm_<jobid>.out

# ── Resource allocation ────────────────────────────────────────────────────
#SBATCH -J "eval emrdm"
#SBATCH --partition=gpu            # TODO: set to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_emrdm_%j.out
#SBATCH --error=logs/eval_emrdm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oywang@caltech.edu

# ── USER SETTINGS ──────────────────────────────────────────────────────────
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase2_emrdm"
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CHECKPOINT="/resnick/groups/perona/oywang/cs159/emrdm_weights/train/sentinel/checkpoints/last.ckpt"  # TODO: path to checkpoint
RESULT_DIR="/scratch/oywang/cs159/results/emrdm_sen12mscr"
CONDA_ENV="/resnick/groups/perona/oywang/conda_envs/myenv"
# ── END USER SETTINGS ──────────────────────────────────────────────────────

set -euo pipefail

# Activate environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${RESULT_DIR}" logs

cd "${CODE_DIR}"

echo "===== Job info ====="
echo "  Node:       $(hostname)"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Code dir:   ${CODE_DIR}"
echo "  Data root:  ${DATA_ROOT}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Result dir: ${RESULT_DIR}"
echo "  Start:      $(date)"
echo "===================="

python eval_emrdm.py \
    --data_root   "${DATA_ROOT}" \
    --checkpoint  "${CHECKPOINT}" \
    --result_dir  "${RESULT_DIR}" \
    --emrdm_dir   "${CODE_DIR}" \
    --split       test \
    --devices     1

echo "===== Finished: $(date) ====="
