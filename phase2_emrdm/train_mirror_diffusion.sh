#!/usr/bin/env bash
# Train a standard (non-residual) EDM in mirror space, conditioned on S1S2.
#
# Prerequisites:
#   1. Phase 1 training must have completed and best.pt must exist at NAMM_CKPT.
#   2. all_train_paths.pkl and all_val_paths.pkl must exist under DATA_ROOT.
#      Run eval_emrdm.py to generate them if missing:
#        python phase2_emrdm/eval_emrdm.py --data_root <DATA_ROOT> --checkpoint <ckpt>
#   3. Run check_mirror_range.sh first and set --sigma_data to the reported
#      global std. Current estimate: 0.08 (from range [-0.174, 0.296]).
#      Confirmed value: std=0.033, mean=0.003 (range [-0.174, 0.296]).
#
# Resume: re-submit this script unchanged after a timeout — the script detects
#   last.pt automatically and resumes from it.
#
# Monitor:  squeue -u $USER
# Output:   tail -f /resnick/groups/perona/oywang/cs159/logs/mirror-edm-train-<jobid>.out

#SBATCH --job-name=mirror-edm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=epyc
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/mirror-edm-train-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/mirror-edm-train-%j.err

source ~/.bashrc
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/resnick/groups/perona/oywang/cs159"
DATA_ROOT="${REPO_ROOT}/data"
PHASE2_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/runs/mirror_edm_final2"

NAMM_CKPT="${REPO_ROOT}/runs/stage3_top/spectral_sw0.1_mw1/best.pt"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Validate prerequisites ────────────────────────────────────────────────────
if [ ! -f "${NAMM_CKPT}" ]; then
    echo "ERROR: Phase 1 checkpoint not found at ${NAMM_CKPT}" >&2
    exit 1
fi
if [ ! -f "${DATA_ROOT}/all_train_paths.pkl" ]; then
    echo "ERROR: all_train_paths.pkl not found under ${DATA_ROOT}" >&2
    echo "       Run eval_emrdm.py to generate split pkl files." >&2
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
conda activate emrdm

# ── Auto-resume if a previous run exists ──────────────────────────────────────
RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/last.pt" ]; then
    echo "=== Found existing last.pt — resuming ==="
    RESUME_ARG="--resume ${OUTPUT_DIR}/last.pt"
fi

# ── Training ──────────────────────────────────────────────────────────────────
echo "===== train_mirror_diffusion: $(date) ====="
echo "  NAMM ckpt : ${NAMM_CKPT}"
echo "  data_root : ${DATA_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"

python "${PHASE2_DIR}/train_mirror_diffusion.py" \
    --data_root         "${DATA_ROOT}"  \
    --namm_ckpt         "${NAMM_CKPT}" \
    --output_dir        "${OUTPUT_DIR}" \
    --n_channels        13              \
    --icnn_layers       6               \
    --icnn_filters      64              \
    --strong_convexity  0.3             \
    --base_ch           64              \
    --depth             4               \
    --emb_dim           256             \
    --sigma_data        0.033           \
    --P_mean            -1.2            \
    --P_std             1.2             \
    --epochs            100             \
    --batch_size        4               \
    --lr                1e-4            \
    --grad_clip         1.0             \
    --fp16                              \
    --num_workers       8               \
    --log_every         100             \
    --wandb                             \
    --wandb_project     cs159           \
    --wandb_run_name    mirror-edm      \
    ${RESUME_ARG}

echo "===== Finished: $(date) ====="
