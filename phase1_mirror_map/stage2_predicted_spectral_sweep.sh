#!/bin/bash
# Spectral-family ablation sweep on EMRDM-predicted inputs (9 configs, one GPU each).
#
# Runs indices 9-17 of run_loss_ablation_predicted.py (the spectral sam×moment
# grid; indices 0-8 are the vgg family and are skipped here).
#
# After the array finishes, collate with:
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> \
#          --partition=gpu --wrap \
#     "source ~/.bashrc && conda activate emrdm && \
#      python $CODE_DIR/run_loss_ablation_predicted.py \
#        --output_root $OUTPUT_ROOT --collate_only \
#        --wandb --wandb_project cs159 --wandb_prefix 'pred-spectral-'"
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, paths below.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --exclude=hpc-93-36
#SBATCH --exclusive
#SBATCH --time=60:00:00

#SBATCH -J "phase1-pred-spectral"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --array=9-17
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-pred-spectral-%A_%a.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-pred-spectral-%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
EMRDM_ROOT="/resnick/groups/perona/oywang/cs159/results"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage2_predicted_spectral_all"

mkdir -p "$OUTPUT_ROOT"

# ── One spectral config per array task (indices 9-17) ────────────────────────
python "$CODE_DIR/run_loss_ablation_predicted.py" \
    --data_root    "$DATA_ROOT" \
    --emrdm_root   "$EMRDM_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       3 \
    --batch_size   8 \
    --lr           2e-4 \
    --num_workers  2 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "pred-spectral-" \
    --config_index "$SLURM_ARRAY_TASK_ID"
