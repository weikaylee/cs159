#!/bin/bash
# Phase 1 loss-ablation sweep as a SLURM array job: one config per array task.
# Default sweep => 18 configs => --array=0-17 (9 vgg style x dis grid +
# 9 spectral sam x moment grid). If you change --sweep_values,
# --vgg_style_values, or --vgg_dis_values, update --array to match
# (M^2 vgg + N^2 spectral configs).
#
# EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, and the paths below.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00            # per array task (one config)

#SBATCH -J "phase1-loss-ablation"
#SBATCH --mail-user=kwei2@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --array=0-17
#SBATCH --output=slurm-loss-ablation-%A_%a.out
#SBATCH --error=slurm-loss-ablation-%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/data"
CODE_DIR="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/phase1_mirror_map"
OUTPUT_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/runs/loss_ablations"

mkdir -p "$OUTPUT_ROOT"

# ── Launch one config (this array task) ──────────────────────────────────────
python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       2 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 \
    --config_index "$SLURM_ARRAY_TASK_ID"

# After all array tasks finish, collate the per-config metrics into
# $OUTPUT_ROOT/ablation_summary.csv (sorted by val_loss):
#
#   python "$CODE_DIR/run_loss_ablation.py" \
#       --output_root "$OUTPUT_ROOT" --collate_only
#
# To chain it automatically after this array job (id printed by sbatch):
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> --partition=gpu --wrap \
#     "source ~/.bashrc && conda activate namm && \
#      python $CODE_DIR/run_loss_ablation.py --output_root $OUTPUT_ROOT --collate_only"
