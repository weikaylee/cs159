#!/usr/bin/env bash
# Scan all mirror_s2 TIFs and report value ranges.
# Reads the mirror dataset written by prepare_mirror_dataset.sh and prints:
#   - any file containing NaN or Inf (direct training NaN source)
#   - global min/max across the whole dataset after /10000 normalisation
#   - 10 most extreme patches (lowest min, highest max)
#
# No GPU needed — pure numpy/rasterio I/O.
#
# Monitor:  squeue -u $USER
# Output:   tail -f /resnick/groups/perona/oywang/cs159/logs/check-mirror-range-<jobid>.out

#SBATCH --job-name=check-mirror-range
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=/resnick/groups/perona/oywang/cs159/logs/check-mirror-range-%j.out
#SBATCH --error=/resnick/groups/perona/oywang/cs159/logs/check-mirror-range-%j.err

source ~/.bashrc
set -euo pipefail

REPO_ROOT="/resnick/groups/perona/oywang/cs159"
DATA_ROOT="${REPO_ROOT}/data"
PHASE2_DIR="${REPO_ROOT}/phase2_emrdm"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

conda activate emrdm

echo "===== check_mirror_range: $(date) ====="
python "${PHASE2_DIR}/check_mirror_range.py" \
    --data_root    "${DATA_ROOT}" \
    --mirror_subdir mirror_s2

echo "===== Finished: $(date) ====="
