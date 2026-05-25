#!/bin/bash
# Stage 2 — Coarse 18-config sweep (15 epochs each, SLURM array).
#
# Runs all 18 ablation configs (9 vgg + 9 spectral) as one array job so every
# config gets its own GPU and they run in parallel.  After the array finishes,
# run the collation step to build ablation_summary.csv and log to wandb:
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> \
#          --partition=gpu --wrap \
#     "source ~/.bashrc && conda activate emrdm && \
#      python $CODE_DIR/run_loss_ablation.py \
#        --output_root $OUTPUT_ROOT --collate_only \
#        --wandb --wandb_project cs159 --wandb_prefix 'stage2-'"
#
# After collation, inspect ablation_summary.csv sorted by val_sam (not just
# val_loss) to pick the top 3 configs for Stage 3.
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, paths below.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=60:00:00

#SBATCH -J "phase1-stage2-coarse"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --array=0-17
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-stage2-%A_%a.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-stage2-%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage2_coarse"

mkdir -p "$OUTPUT_ROOT"

# ── One config per array task ────────────────────────────────────────────────
python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       3 \
    --batch_size   32 \
    --lr           4e-4 \
    --num_workers  2 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "stage2-" \
    --config_index "$SLURM_ARRAY_TASK_ID"
