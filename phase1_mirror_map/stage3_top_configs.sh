#!/bin/bash
# Stage 3 — Top configs from Stage 2 at full epochs (100).
#
# After Stage 2 collation, pick the top 3 configs by val_sam from
# ablation_summary.csv (not just val_loss — spectral consistency matters).
# Edit TOP_CONFIGS below with the config names from that file.
#
# Config name format examples:
#   vgg_st100_ds1        (style_weight=100, dis_weight=1)
#   vgg_st1000_ds10      (style_weight=1000, dis_weight=10)
#   spectral_sw1_mw1     (sam_weight=1, moment_weight=1)
#   spectral_sw0.1_mw10  (sam_weight=0.1, moment_weight=10)
#
# The three configs run in parallel, one per GPU.  If only 2 GPUs are
# available, remove GPU2 / CONFIG3 and request --gres=gpu:2 / --ntasks=2.
#
# TODO EDIT BEFORE SUBMITTING: TOP_CONFIGS, --epochs, --mail-user, paths.

#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
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
OUTPUT_ROOT="/resnick/groups/perona/oywang/cs159/runs/stage3_top"

mkdir -p "$OUTPUT_ROOT"

# ── EDIT: top 3 config names from Stage 2 ablation_summary.csv ──────────────
CONFIG1="EDIT_ME"   # e.g. vgg_st100_ds1
CONFIG2="EDIT_ME"   # e.g. spectral_sw1_mw1
CONFIG3="EDIT_ME"   # e.g. vgg_st1000_ds10

# ── Run three configs in parallel, one GPU each ──────────────────────────────
CUDA_VISIBLE_DEVICES=0 python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       100 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "stage3-" \
    --only         "$CONFIG1" &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       100 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "stage3-" \
    --only         "$CONFIG2" &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python "$CODE_DIR/run_loss_ablation.py" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --roi          all \
    --epochs       100 \
    --batch_size   16 \
    --lr           2e-4 \
    --num_workers  8 \
    --max_sigma    0.1 \
    --fp16 \
    --wandb --wandb_project cs159 --wandb_prefix "stage3-" \
    --only         "$CONFIG3" &
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
python "$CODE_DIR/run_loss_ablation.py" \
    --output_root  "$OUTPUT_ROOT" \
    --collate_only \
    --wandb --wandb_project cs159 --wandb_prefix "stage3-"

[ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] && [ $EXIT3 -eq 0 ]
