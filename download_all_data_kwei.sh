#!/bin/bash
#SBATCH -J "download SEN12MS-CR dataset"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=kayleewei023@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/download_%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159/download_%j.err

set -euo pipefail

REPO_ROOT="/resnick/groups/CS156b/from_central/2026/SSSClassCSNerds/cs159"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emrdm

cd "${REPO_ROOT}"

echo "===== Job info ====="
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "===================="

python download_all_data.py

echo "===== Finished: $(date) ====="
