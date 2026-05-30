#!/bin/bash
# Stage 3 — Top 3 configs from Stage 2, warm-started from stage2 best.pt.
#
# Configs (ranked by val_sam from ablation_summary.csv):
#   spectral_sw0.1_mw1   val_sam=0.0617  val_psnr=37.30  val_ssim=0.944  (best overall)
#   spectral_sw1_mw10    val_sam=0.0598  val_psnr=33.53  val_ssim=0.944
#   spectral_sw10_mw10   val_sam=0.0580  val_psnr=30.56  val_ssim=0.910  (best SAM)
#
# Calls train_mirror_map.py directly (not run_loss_ablation.py) so --resume
# can be passed. run_loss_ablation.py --collate_only at the end builds the
# stage3 ablation_summary.csv.
#
# The three configs run in parallel, one per GPU.  If only 2 GPUs are
# available, remove GPU2 / CONFIG3 and request --gres=gpu:2 / --ntasks=2.
#
# TODO EDIT BEFORE SUBMITTING: --epochs, --mail-user, paths.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

#SBATCH -J "phase1-stage3-top"
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/slurm-stage3-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/slurm-stage3-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate emrdm

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CODE_DIR="/resnick/groups/perona/oywang/cs159/phase1_mirror_map"
STAGE2_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage2_coarse"
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage3_top"

mkdir -p "$OUTPUT_ROOT"

# ── Top 3 configs from Stage 2 (by val_sam) ──────────────────────────────────
CONFIG1="spectral_sw0.1_mw1"
CONFIG2="spectral_sw1_mw10"
CONFIG3="spectral_sw10_mw10"

# ── Shared training args ──────────────────────────────────────────────────────
COMMON=(
    --data_root   "$DATA_ROOT"
    --roi         all
    --epochs      100
    --batch_size  16
    --lr          2e-4
    --num_workers 4
    --max_sigma   0.1
    --fp16
    --wandb --wandb_project cs159
)

# ── Run three configs in parallel, one GPU each ──────────────────────────────
CUDA_VISIBLE_DEVICES=0 python "$CODE_DIR/train_mirror_map.py" \
    "${COMMON[@]}" \
    --output_dir     "$OUTPUT_ROOT/$CONFIG1" \
    --resume         "$STAGE2_ROOT/$CONFIG1/best.pt" \
    --dis_weight 0 --style_weight 0 \
    --sam_weight     0.1 --moment_weight 1 \
    --wandb_run_name "stage3-$CONFIG1" &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python "$CODE_DIR/train_mirror_map.py" \
    "${COMMON[@]}" \
    --output_dir     "$OUTPUT_ROOT/$CONFIG2" \
    --resume         "$STAGE2_ROOT/$CONFIG2/best.pt" \
    --dis_weight 0 --style_weight 0 \
    --sam_weight     1 --moment_weight 10 \
    --wandb_run_name "stage3-$CONFIG2" &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python "$CODE_DIR/train_mirror_map.py" \
    "${COMMON[@]}" \
    --output_dir     "$OUTPUT_ROOT/$CONFIG3" \
    --resume         "$STAGE2_ROOT/$CONFIG3/best.pt" \
    --dis_weight 0 --style_weight 0 \
    --sam_weight     10 --moment_weight 10 \
    --wandb_run_name "stage3-$CONFIG3" &
PID3=$!

# ── Wait and propagate failures ──────────────────────────────────────────────
wait $PID1; EXIT1=$?
wait $PID2; EXIT2=$?
wait $PID3; EXIT3=$?

for pair in "$EXIT1:$CONFIG1" "$EXIT2:$CONFIG2" "$EXIT3:$CONFIG3"; do
    code="${pair%%:*}"; name="${pair##*:}"
    [ "$code" -ne 0 ] && echo "ERROR: $name failed (exit $code)" >&2
done

# ── Combined collation + wandb summary ───────────────────────────────────────
# Collates the 3 configs; other ablation configs show as "missing" (expected).
python "$CODE_DIR/run_loss_ablation.py" \
    --output_root  "$OUTPUT_ROOT" \
    --collate_only \
    --wandb --wandb_project cs159 --wandb_prefix "stage3-"

[ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] && [ $EXIT3 -eq 0 ]
