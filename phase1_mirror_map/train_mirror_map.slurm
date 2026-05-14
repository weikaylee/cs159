#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=5   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --partition=gpu   # memory per CPU core

#SBATCH --time=48:00:00   # walltime format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH -J "densenet121 pneu 1.05 contrast 30 epochs"   # job name
#SBATCH --mail-user=oywang@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

# ADJUST THIS!!!!
# /home/oywang/miniconda3/envs/WowCIT/bin/python olivia0522pneutrans3.py

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm    # re-use the EMRDM env: torch, rasterio, torchvision

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159"
CODE_DIR="/resnick/groups/perona/oywang/cs159/code/phase1_mirror_map"
OUTPUT_DIR="/resnick/groups/perona/oywang/cs159/checkpoints/phase1"
LOG_DIR="/resnick/groups/perona/oywang/cs159/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Launch ───────────────────────────────────────────────────────────────────
python "$CODE_DIR/train_mirror_map.py" \
    --data_root    "$DATA_ROOT" \
    --output_dir   "$OUTPUT_DIR" \
    --roi          ROIs1158_spring \
    --n_channels   13 \
    --ngf          64 \
    --n_res_blocks 6 \
    --icnn_filters 64 \
    --icnn_layers  6 \
    --strong_convexity 0.3 \
    --epochs       100 \
    --batch_size   16 \
    --lr           2e-4 \
    --max_sigma    0.1 \
    --style_weight 100.0 \
    --cycle_weight 1.0 \
    --constr_weight 1.0 \
    --reg_weight   0.001 \
    --ema_rate     0.999 \
    --num_workers  8 \
    --log_every    50 \
    --fp16

# To resume from a checkpoint, add:
#   --resume "$OUTPUT_DIR/last.pt"
# To stream metrics to Weights & Biases (online), add:
#   --wandb --wandb_project cs159
