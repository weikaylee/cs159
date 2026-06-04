#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

#SBATCH --time=12:00:00   # eval is shorter than training; adjust if needed

#SBATCH -J "mirror-edm-eval-dropout"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/mirror-edm-eval-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/mirror-edm-eval-%j.err


# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm    # re-use the EMRDM env: torch, rasterio, torchvision

# ── Paths — ADJUST THESE ─────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase2_emrdm"
NAMM_CKPT="/resnick/groups/perona/oywang/cs159/runs/stage3_top/spectral_sw0.1_mw1/best.pt"
EDM_CKPT="/resnick/groups/perona/oywang/cs159/runs/mirror_edm_final_dropout_wcloudweight/best.pt"
OUTPUT_DIR="/resnick/groups/perona/oywang/cs159/output/mirror_diffusion_eval_dropout_wcloudweight"

mkdir -p "$OUTPUT_DIR"

# ── Launch ───────────────────────────────────────────────────────────────────
python "$CODE_DIR/run_mirror_diffusion.py" \
    --data_root    "$DATA_ROOT" \
    --namm_ckpt    "$NAMM_CKPT" \
    --edm_ckpt     "$EDM_CKPT" \
    --output_dir   "$OUTPUT_DIR" \
    --n_channels   13 \
    --ngf          64 \
    --n_res_blocks 6 \
    --base_ch      64 \
    --depth        4 \
    --emb_dim      256 \
    --sigma_data   0.033 \
    --sampler      heun \
    --steps        40 \
    --sigma_min    0.002 \
    --sigma_max    5.0 \
    --rho          7.0 \
    --split        test \
    --batch_size   4 \
    --num_workers  4 \
    --fp16 \
    && \
python "$CODE_DIR/visualize_mirror_diffusion.py" \
    --output_dir "$OUTPUT_DIR" \
    --data_root  "$DATA_ROOT" \
    --n_samples  12 \
    --select     spread \
    --bands      rgb

# To also save raw mirror-space predictions, add to run_mirror_diffusion.py:
#   --save_mirror
# To evaluate on val split instead of test, change:
#   --split val
# To generate SWIR false-colour visualisation instead of natural RGB, change:
#   --bands swir


# --max_samples   12 \

