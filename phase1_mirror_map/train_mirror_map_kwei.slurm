#!/usr/bin/env bash
# Train NAMM mirror maps (Phase 1) with EMRDM paired loss — kwei credentials.
# Runs all three sam_weight × moment_weight configs as a SLURM array job.
#
# Submit:  sbatch train_mirror_map_kwei.slurm
# Monitor: tail -f logs/train_kwei_<JOBID>_<TASKID>.out

#SBATCH -J "train-mirror-kwei"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/train_kwei_%A_%a.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/train_kwei_%A_%a.err

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159"
DATA_ROOT="${REPO_ROOT}/data"
EMRDM_ROOT="${REPO_ROOT}/data/emrdm"
RUNS_DIR="${REPO_ROOT}/runs/stage3_kwei"
CODE_DIR="${REPO_ROOT}/phase1_mirror_map"

# ── Config arrays (indexed by SLURM_ARRAY_TASK_ID) ──────────────────────────
# Task 0: sw1_mw0.1  — known-good config
# Task 1: sw1_mw1    — known-good config
# Task 2: sw0.1_mw0.1 — previously caused NaN (fixed by clip_weights + nan guard)
SAM_WEIGHTS=(1.0 1.0 0.1)
MOMENT_WEIGHTS=(0.1 1.0 0.1)
RUN_NAMES=("kwei-sw1_mw0.1" "kwei-sw1_mw1" "kwei-sw0.1_mw0.1")

IDX=${SLURM_ARRAY_TASK_ID}
SW=${SAM_WEIGHTS[$IDX]}
MW=${MOMENT_WEIGHTS[$IDX]}
RUN=${RUN_NAMES[$IDX]}
OUTPUT_DIR="${RUNS_DIR}/${RUN}"

# ── Environment ──────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emrdm

mkdir -p "${OUTPUT_DIR}" "${CODE_DIR}/logs"

echo "===== Job info ====="
echo "  Node:          $(hostname)"
echo "  Array task:    ${IDX} (${RUN})"
echo "  sam_weight:    ${SW}"
echo "  moment_weight: ${MW}"
echo "  Data root:     ${DATA_ROOT}"
echo "  EMRDM root:    ${EMRDM_ROOT}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Start:         $(date)"
echo "===================="

cd "${CODE_DIR}"

python train_mirror_map.py \
    --data_root       "${DATA_ROOT}" \
    --emrdm_root      "${EMRDM_ROOT}" \
    --roi all \
    --epochs 10 \
    --batch_size 8 \
    --lr 5e-5 \
    --emrdm_loss_weight 0.3 \
    --num_workers 4 \
    --max_sigma 0.1 \
    --fp16 \
    --dis_weight 0 \
    --style_weight 0 \
    --sam_weight      "${SW}" \
    --moment_weight   "${MW}" \
    --wandb \
    --wandb_project cs159 \
    --output_dir      "${OUTPUT_DIR}" \
    --wandb_run_name  "${RUN}"

echo "===== Finished: $(date) ====="
