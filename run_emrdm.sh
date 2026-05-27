#!/bin/bash
#SBATCH --job-name=emrdm_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/emrdm_infer.%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/emrdm_infer.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oywang@caltech.edu

set -euo pipefail

ROOT_DIR="/resnick/groups/perona/oywang/"
CS159_DIR="${ROOT_DIR}/cs159"

cd "${CS159_DIR}"

# Activate conda env (kwei credentials)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emrdm_test

# Optional overrides
# export EMRDM_REPO="${ROOT_DIR}/EMRDM"
# export EMRDM_DATA_ROOT="${CS159_DIR}/data"
# export EMRDM_CKPT="${CS159_DIR}/last.ckpt"
# export EMRDM_RESULTS_DIR="${ROOT_DIR}/results"
# export EMRDM_DEVICES="1"
# export EMRDM_INSTALL_DEPS="0"
# export EMRDM_DOWNLOAD_DATA="0"
# export EMRDM_DOWNLOAD_WEIGHTS="0"
# export EMRDM_PATCH_NATTEN="1"

if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
	export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

srun --ntasks=1 --gpus=1 python model.py
