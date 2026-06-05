#!/usr/bin/env bash
# Run EMRDM inference on cs159/data using cs159/last.ckpt.
# Outputs go to the main project's data/emrdm directory.
#
# Submit with:  sbatch run_emrdm_predict_kwei.sh
# Monitor:      tail -f logs/emrdm_kwei_<jobid>.out

#SBATCH -J "emrdm-predict-kwei"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=cs159/logs/phase2-emrdm_kwei_%j.out
#SBATCH --error=cs159/logs/phase2-emrdm_kwei_%j.err

set -euo pipefail

REPO_ROOT="/resnick/groups/perona/oywang/cs159"
DATA_ROOT="${REPO_ROOT}/data"
CKPT_PATH="${REPO_ROOT}/emrdm_weights/train/sentinel/checkpoints/last.ckpt"
OUTPUT_DIR="/resnick/groups/perona/oywang/cs159/data/emrdm"
CODE_DIR="${REPO_ROOT}/phase2_emrdm"

echo "===== Job info ====="
echo "  Node:       $(hostname)"
echo "  Data root:  ${DATA_ROOT}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Start:      $(date)"
echo "===================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emrdm_test

mkdir -p "${OUTPUT_DIR}" "${CODE_DIR}/logs"

cd "${CODE_DIR}"

python run_emrdm_predict.py \
    --data_root  "${DATA_ROOT}" \
    --ckpt_path  "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --skip_pkl

echo "===== Finished: $(date) ====="
