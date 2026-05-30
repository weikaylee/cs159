#!/usr/bin/env bash
#SBATCH -J "emrdm_inference"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/emrdm_%j.out
#SBATCH --error=logs/emrdm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oywang@caltech.edu

# ── USER SETTINGS ────────────────────────────────────────────────────────
EMRDM_DIR="/resnick/groups/perona/oywang/cs159/phase2_emrdm/cs159/phase2_emrdm/EMRDM"
DATA_ROOT="/resnick/groups/perona/oywang/cs159/data"
CKPT="/resnick/groups/perona/oywang/cs159/emrdm_weights/train/sentinel/checkpoints/last.ckpt"
CONDA_ENV="/resnick/groups/perona/oywang/conda_envs/myenv"
# ── END USER SETTINGS ────────────────────────────────────────────────────

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p logs

cd "${EMRDM_DIR}"

# Edit sentinel.yaml to point to correct paths
python - <<EOF
with open("configs/example_training/sentinel.yaml") as f:
    content = f.read()
content = content.replace(
    '# ckpt_path: "" # your checkpoint path',
    'ckpt_path: "${CKPT}"'
)
content = content.replace(
    'root: "/remote-home/share/dmb_nas2/liuyi/SEN12MSCR"',
    'root: "${DATA_ROOT}"'
)
content = content.replace('devices: 2,4', 'devices: "1"')
with open("configs/example_training/sentinel_hpc.yaml", "w") as f:
    f.write(content)
print("yaml written")
EOF

echo "===== Job info ====="
echo "  Node:     $(hostname)"
echo "  EMRDM:    ${EMRDM_DIR}"
echo "  Data:     ${DATA_ROOT}"
echo "  Ckpt:     ${CKPT}"
echo "  Start:    $(date)"
echo "===================="

python main.py \
    --base configs/example_training/sentinel_hpc.yaml \
    --enable_tf32 \
    -t false \
    --no-test true \
    --predict true

echo "===== Finished: $(date) ====="