#!/bin/bash

# SLURM GPU Script: Run test_train_mirror_map.py integration test with GPU
#SBATCH --job-name=test_mirror_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=test_train_mirror_map_%j.log
#SBATCH --error=test_train_mirror_map_%j.err

set -e  # Exit on error

# Initialize conda
source ~/.bashrc

# Activate GPU-enabled environment (CUDA 12.1)
# TODO change this
source /home/kwei2/miniforge3/etc/profile.d/conda.sh
conda activate emrdm

# Run the integration test with pytest
python -m pytest tests/integration/test_train_mirror_map.py -v
