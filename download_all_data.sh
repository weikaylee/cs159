#!/bin/bash
#SBATCH -J "download SEN12MS-CR dataset"   # job name
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=5   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:0
#SBATCH --mem=256G                   # Low memory, just enough to unpack files
#SBATCH --time=24:00:00              # 2 hours (adjust based on dataset size)
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=download_%j.out 

# Run your original script or commands
/resnick/groups/perona/oywang/conda_envs/myenv/bin/python download_all_data.py
