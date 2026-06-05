#!/usr/bin/env bash
# Train EDM in raw reflectance space, conditioned on S1S2.
# No Phase 1 NAMM checkpoint required.
#
# Prerequisites:
#   all_train_paths.pkl and all_val_paths.pkl must exist under DATA_ROOT.
#   (Already present at /resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/data)
#   Non-ROI subdirs (emrdm/, mirror_s2/, etc.) are automatically ignored by
#   build_triplets — only directories matching ROIs*_*_s{1,2}* are indexed.
#
# Resume: re-submit this script unchanged after a timeout — the script detects
#   last.pt automatically and resumes from it.
#
# Monitor:  squeue -u kwei2
# Output:   tail -f <LOG_DIR>/raw-edm-train-<jobid>.out

#SBATCH --job-name=raw-edm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/raw-edm-train-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/raw-edm-train-%j.err

source ~/.bashrc
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159"
DATA_ROOT="${REPO_ROOT}/data"
PHASE2_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159"
OUTPUT_DIR="${REPO_ROOT}/runs/raw_edm"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Validate prerequisites ────────────────────────────────────────────────────
if [ ! -f "${DATA_ROOT}/all_train_paths.pkl" ]; then
    echo "ERROR: all_train_paths.pkl not found under ${DATA_ROOT}" >&2
    echo "       Run eval_emrdm.py to generate split pkl files." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emrdm

# ── Auto-resume if a previous run exists ──────────────────────────────────────
RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/last.pt" ]; then
    echo "=== Found existing last.pt — resuming ==="
    RESUME_ARG="--resume ${OUTPUT_DIR}/last.pt"
fi

# ── Training ──────────────────────────────────────────────────────────────────
echo "===== train_raw_diffusion: $(date) ====="
echo "  data_root : ${DATA_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"

python "${PHASE2_DIR}/train_raw_diffusion.py" \
    --data_root         "${DATA_ROOT}"  \
    --output_dir        "${OUTPUT_DIR}" \
    --n_channels        13              \
    --base_ch           64              \
    --depth             4               \
    --emb_dim           256             \
    --sigma_data        0.091           \
    --P_mean            -1.2            \
    --P_std             1.2             \
    --epochs            100             \
    --batch_size        4               \
    --lr                1e-4            \
    --grad_clip         1.0             \
    --s2_dropout        0.15            \
    --fp16                              \
    --num_workers       8               \
    --log_every         100             \
    --wandb                             \
    --wandb_project     baseline        \
    --wandb_entity      cs159           \
    --wandb_run_name    raw-edm         \
    ${RESUME_ARG}

echo "===== Finished: $(date) ====="
