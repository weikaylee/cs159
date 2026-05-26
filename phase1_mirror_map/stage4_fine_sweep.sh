#!/bin/bash
# Stage 4 — Fine sweep around the Stage 3 winner (100 epochs, SLURM array).
#
# Set FAMILY to the winning family from Stage 3 before submitting:
#
#   FAMILY=vgg       → sweeps style ∈ {50,100,200} × dis ∈ {0.5,1,2} (9 configs, indices 0-8)
#   FAMILY=spectral  → sweeps sam ∈ {0.5,1,2} × moment ∈ {0.5,1,2}  (9 configs, indices 9-17)
#
# The sweep values are centred on the Stage 3 winner.  Adjust
# --vgg_style_values / --vgg_dis_values / --sweep_values below if the winner
# sits at the edge of those ranges (shift the window accordingly).
#
# After the array finishes, run the collation + wandb step:
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> \
#          --partition=gpu --wrap \
#     "source ~/.bashrc && conda activate emrdm && \
#      python $CODE_DIR/run_loss_ablation.py \
#        --output_root $OUTPUT_ROOT \
#        --vgg_style_values 50,100,200 --vgg_dis_values 0.5,1,2 \
#        --sweep_values 0.5,1,2 \
#        --collate_only --wandb --wandb_project cs159 --wandb_prefix 'stage4-'"
#
# TODO EDIT BEFORE SUBMITTING: FAMILY, --mail-user, --partition, paths below.

# ── EDIT: set to 'vgg' or 'spectral' based on Stage 3 winner ────────────────
FAMILY="EDIT_ME"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

#SBATCH -J "phase1-stage4-fine"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-stage4-%A_%a.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-stage4-%A_%a.err

# Submit with the correct --array for your family:
#   vgg family:       sbatch --array=0-8  stage4_fine_sweep.sh
#   spectral family:  sbatch --array=9-17 stage4_fine_sweep.sh
#SBATCH --array=0-8

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage4_fine"

mkdir -p "$OUTPUT_ROOT"

# ── Select sweep args for the winning family ─────────────────────────────────
if [ "$FAMILY" = "vgg" ]; then
    SWEEP_ARGS="--vgg_style_values 50,100,200 --vgg_dis_values 0.5,1,2"
elif [ "$FAMILY" = "spectral" ]; then
    SWEEP_ARGS="--sweep_values 0.5,1,2"
else
    echo "ERROR: FAMILY must be 'vgg' or 'spectral', got '${FAMILY}'" >&2
    exit 1
fi

# ── One config per array task ────────────────────────────────────────────────
# shellcheck disable=SC2086
python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       100 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "stage4-${FAMILY}-" \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    $SWEEP_ARGS
