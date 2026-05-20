#!/bin/bash
#SBATCH -J "download SEN12MS-CR dataset"   # job name
#SBATCH --partition=cpu          # Uses standard CPU nodes, much faster queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # Low memory, just enough to unpack files
#SBATCH --time=24:00:00              # 2 hours (adjust based on dataset size)
#SBATCH --mail-user=oywang@caltech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=download_%j.out 

# Run your original script or commands
/resnick/groups/perona/oywang/conda_envs/myenv/bin/python download_all_data.py
