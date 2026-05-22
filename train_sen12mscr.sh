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
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1               # TODO: adjust GPU count (must match batch_size / per_device_batch)
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
OUTPUT_DIR="/scratch/oywang/cs159/checkpoints/namm_sen12mscr/job_${SLURM_JOB_ID}"
CONDA_ENV="/resnick/groups/perona/oywang/conda_envs/myenv"      # TODO: conda env name
# ── END USER SETTINGS ──────────────────────────────────────────────────────

set -euo pipefail

# Do NOT load the system CUDA module — jax[cuda12] ships its own CUDA 12.9
# libraries via pip and is fully self-contained.  Mixing system CUDA 12.1 with
# pip CUDA 12.9 causes CUDA_ERROR_UNKNOWN at StreamExecutor initialization.

# Activate environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Redirect large caches to scratch to avoid filling home directory.
export HF_HOME=/resnick/groups/perona/oywang/.cache/huggingface
export TORCH_HOME=/resnick/groups/perona/oywang/.cache/torch
export PIP_CACHE_DIR=/resnick/groups/perona/oywang/.cache/pip

# Add all pip-installed NVIDIA CUDA library directories to LD_LIBRARY_PATH so
# JAX's PJRT plugin can find cublas, cusparse, cusolver, cufft, etc.
NVIDIA_BASE="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
for lib_path in "${NVIDIA_BASE}"/*/lib; do
  [ -d "${lib_path}" ] && export LD_LIBRARY_PATH="${lib_path}:${LD_LIBRARY_PATH:-}"
done
# Expose the saved cuDNN 9 so JAX can find libcudnn.so.9.
export LD_LIBRARY_PATH="/resnick/groups/perona/oywang/cudnn9/lib:${LD_LIBRARY_PATH:-}"
# Allow XLA to use a fallback cuDNN convolution algorithm when autotuning fails.
# Required when cuDNN 8.9 is loaded at runtime instead of the cuDNN 9 JAX expects.
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi: no GPU visible"
python -c "
import jax
try:
    print('JAX devices:', jax.devices())
except Exception as e:
    print('JAX devices error:', e)
"

python train_namm.py \
    --config                  configs/sen12mscr_config.py \
    --workdir                 "${OUTPUT_DIR}" \
    --data_root               "${DATA_ROOT}" \
    --wandb \
    --wandb_project           cs159 \
    --wandb_run_name          "namm_sen12mscr_job${SLURM_JOB_ID}" \
    --config.training.batch_size=16 \
    --config.training.n_epochs=100 \
    --config.optim.learning_rate=2e-4 \
    --config.optim.constraint_weight=0.1 \
    --config.constraint.style_weight=100.0 \
    --config.data.num_workers=0

echo "===== Finished: $(date) ====="
