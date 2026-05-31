# TODO: download namm model checkpoint and update NAMM_CKPT 
#!/bin/bash

#SBATCH --job-name=prepare_mirror_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/prep-mirror-dataset-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/prep-mirror-dataset-%j.err

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/resnick/groups/perona/oywang/cs159"
DATA_ROOT="${REPO_ROOT}/data"
PHASE1_DIR="${REPO_ROOT}/phase1_mirror_map"
PHASE2_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

# ── Step 1: Use pre-trained NAMM g_phi checkpoint ─────────────────────────────
NAMM_CKPT="${REPO_ROOT}/runs/stage3_top/spectral_sw0.1_mw1/best.pt"
if [ ! -f "${NAMM_CKPT}" ]; then
    echo "ERROR: checkpoint not found at ${NAMM_CKPT}" >&2
    exit 1
fi
echo "=== Step 1: Using g_phi checkpoint at ${NAMM_CKPT} ==="

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
    --seed             42 \
    --overwrite
