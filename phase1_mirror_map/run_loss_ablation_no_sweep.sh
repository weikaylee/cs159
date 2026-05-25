#!/bin/bash
# Phase 1 loss-ablation: best VGG and best Spectral config only (single jobs, not array)
# This script runs one VGG-style and one Spectral-style config with optimal settings.
#
# Time estimate: observed ~6 s/step at 256×256 with NaN training (no weight updates).
# With real gradient flow at --patch_size 64 (NAMM official resolution, ~8-10x faster
# spatially), expect ~1-2 s/step → ~25 epochs in ~24 h per config.  Both configs run
# in parallel on separate GPUs, so wall time equals one config's training time.
# If step time differs significantly, adjust --epochs and --time accordingly.
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, --partition, and the paths below.

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -J "phase1-loss-ablation-best"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-loss-ablation-best-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-loss-ablation-best-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/loss_ablations_no_sweep"
mkdir -p "$OUTPUT_ROOT"

# ── Best VGG config ─────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT/vgg" \
    --roi          all \
    --epochs       25 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --patch_size   64 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "ablation-vgg-" \
    --style_weight 100 \
    --dis_weight   1 \
    --sam_weight   0 \
    --moment_weight 0 &
VGG_PID=$!

# ── Best Spectral config ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT/spectral" \
    --roi          all \
    --epochs       25 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --patch_size   64 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "ablation-spectral-" \
    --sam_weight   1 \
    --moment_weight 1 \
    --style_weight 0 \
    --dis_weight   0 &
SPECTRAL_PID=$!

# ── Wait for both and propagate failures ────────────────────────────────────
wait $VGG_PID
VGG_EXIT=$?
wait $SPECTRAL_PID
SPECTRAL_EXIT=$?
if [ $VGG_EXIT -ne 0 ]; then
    echo "ERROR: VGG job failed with exit code $VGG_EXIT" >&2
fi
if [ $SPECTRAL_EXIT -ne 0 ]; then
    echo "ERROR: Spectral job failed with exit code $SPECTRAL_EXIT" >&2
fi

# Exit non-zero if either job failed (triggers SLURM FAIL mail)
[ $VGG_EXIT -eq 0 ] && [ $SPECTRAL_EXIT -eq 0 ]