#!/bin/bash
# Stage 3 (predicted) — retry for spectral_sw0.1_mw0.1 only.
# Warm-started from stage2_predicted_spectral best.pt, trained on EMRDM-predicted inputs.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

#SBATCH -J "phase1-pred-stage3-sw0.1-mw0.1"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-pred-stage3-retry-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-pred-stage3-retry-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
EMRDM_ROOT="/resnick/groups/perona/oywang/cs159/results"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
STAGE2_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage2_predicted_spectral_all"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage3_predicted_top"

mkdir -p "$OUTPUT_ROOT"

CONFIG="spectral_sw0.1_mw0.1"

python "$CODE_DIR/train_mirror_map.py" \
    --data_root    "$DATA_ROOT" \
    --emrdm_root   "$EMRDM_ROOT" \
    --roi          all \
    --epochs       100 \
    --batch_size   8 \
    --lr           5e-5 \
    --emrdm_loss_weight 0.3 \
    --num_workers  4 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 \
    --output_dir     "$OUTPUT_ROOT/$CONFIG" \
    --resume         "$STAGE2_ROOT/$CONFIG/best.pt" \
    --dis_weight 0 --style_weight 0 \
    --sam_weight     0.1 --moment_weight 0.1 \
    --wandb_run_name "pred-stage3-$CONFIG"
