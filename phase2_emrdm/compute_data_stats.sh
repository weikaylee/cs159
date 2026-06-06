#!/usr/bin/env bash
# Compute mean and std of clean S2 reflectance over the training split.
# The printed global_std is the correct --sigma_data value for train_raw_diffusion.
#
# Output: logs/compute-data-stats-<jobid>.out
# After completion, grep for sigma_data:
#   grep "sigma_data" logs/compute-data-stats-*.out

#SBATCH --job-name=compute-data-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/logs/compute-data-stats-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/logs/compute-data-stats-%j.err

source ~/.bashrc
set -euo pipefail

REPO_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159"
DATA_ROOT="${REPO_ROOT}/data"
LOG_DIR="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/logs"

mkdir -p "${LOG_DIR}"

source /home/kwei2/miniforge3/etc/profile.d/conda.sh
conda activate cs156b

echo "===== compute_data_stats: $(date) ====="
echo "  data_root: ${DATA_ROOT}"

python "${REPO_ROOT}/phase2_emrdm/compute_data_stats.py" \
    --data_root "${DATA_ROOT}"

echo "===== Finished: $(date) ====="
