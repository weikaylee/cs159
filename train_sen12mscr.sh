#!/usr/bin/env bash
# SLURM job script for training NAMM on SEN12MS-CR.
#
# Edit the four USER SETTINGS lines before submitting, then:
#   sbatch scripts/train_sen12mscr.sh
#
# Monitor:  squeue -u $USER
# Output:   tail -f logs/sen12mscr_<jobid>.out

# ── Resource allocation ────────────────────────────────────────────────────
#SBATCH -J "train mirror map"   # job name
#SBATCH --partition=gpu            # TODO: set to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4               # TODO: adjust GPU count (must match batch_size / per_device_batch)
#SBATCH --cpus-per-task=16         # DataLoader workers × GPUs
#SBATCH --mem=128G                  # TODO: increase if OOM
#SBATCH --time=48:00:00            # TODO: adjust wall-clock limit
#SBATCH --output=logs/sen12mscr_%j.out
#SBATCH --error=logs/sen12mscr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oywang@caltech.edu   # TODO: set your email

# ── USER SETTINGS ──────────────────────────────────────────────────────────
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map/namm"
   # TODO: absolute path to namm/
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"                                  # TODO: SEN12MS-CR root on HPC
OUTPUT_DIR="/scratch/oywang/cs159/checkpoints/namm_sen12mscr"
CONDA_ENV="/resnick/groups/perona/oywang/conda_envs/myenv"      # TODO: conda env name
# ── END USER SETTINGS ──────────────────────────────────────────────────────

set -euo pipefail

# Activate environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}" logs

cd "${CODE_DIR}"

echo "===== Job info ====="
echo "  Node:       $(hostname)"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Code dir:   ${CODE_DIR}"
echo "  Data root:  ${DATA_ROOT}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Start:      $(date)"
echo "===================="

python train_namm.py \
    --config              configs/sen12mscr_config.py \
    --workdir             "${OUTPUT_DIR}" \
    --data_root           "${DATA_ROOT}" \
    --wandb \
    --wandb_project       cs159 \
    --wandb_run_name      "namm_sen12mscr_$(date +%Y%m%d_%H%M%S)" \
    --config.training.batch_size=16 \
    --config.training.n_epochs=100 \
    --config.optim.learning_rate=2e-4 \
    --config.optim.constraint_weight=0.1 \
    --config.constraint.style_weight=100.0

echo "===== Finished: $(date) ====="
