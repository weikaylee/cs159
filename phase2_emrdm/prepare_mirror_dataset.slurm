#!/bin/bash

#SBATCH --job-name=prepare_mirror_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

#SBATCH --mail-user=kayleewei023@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolve repo root from script location (phase2_emrdm/../)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
PHASE1_DIR="${REPO_ROOT}/phase1_mirror_map"
PHASE2_DIR="${REPO_ROOT}/phase2_emrdm"
DUMMY_CKPT_DIR="${REPO_ROOT}/checkpoints/dummy_namm"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${DUMMY_CKPT_DIR}" "${LOG_DIR}"

# ── Step 1: Train dummy NAMM g_phi (1 epoch, small settings) ──────────────────
echo "=== Step 1: Training dummy NAMM g_phi (1 epoch) ==="
python "${PHASE1_DIR}/train_mirror_map.py" \
    --data_root      "${DATA_ROOT}" \
    --output_dir     "${DUMMY_CKPT_DIR}" \
    --roi            ROIs1158_spring \
    --n_channels     13 \
    --icnn_filters   64 \
    --icnn_layers    6 \
    --strong_convexity 0.3 \
    --epochs         1 \
    --batch_size     4 \
    --patch_size     64 \
    --lr             2e-4 \
    --num_workers    4 \
    --log_every      10 \
    --fp16

NAMM_CKPT="${DUMMY_CKPT_DIR}/best.pt"
if [ ! -f "${NAMM_CKPT}" ]; then
    echo "ERROR: checkpoint not found at ${NAMM_CKPT}" >&2
    exit 1
fi

# ── Step 2: Generate mirror-space targets ─────────────────────────────────────
echo "=== Step 2: Running prepare_mirror_dataset ==="
python "${PHASE2_DIR}/prepare_mirror_dataset.py" \
    --data_root        "${DATA_ROOT}" \
    --namm_ckpt        "${NAMM_CKPT}" \
    --mirror_subdir    mirror_s2 \
    --device           cuda \
    --n_channels       13 \
    --icnn_filters     64 \
    --icnn_layers      6 \
    --strong_convexity 0.3 \
    --seed             42
