#!/usr/bin/env python3
"""
EMRDM inference script for DTU HPC.
Run from ~/emrdm_workspace after setup.

Usage:
    python run_emrdm_dtu.py
"""

import os
import subprocess
import sys
import pickle
import glob

# ============ SETTINGS ============
HOME = os.path.expanduser("~")
WORKSPACE = os.path.join(HOME, "emrdm_workspace")
DATA_ROOT = os.path.join(WORKSPACE, "cs159/data")
CKPT = os.path.join(WORKSPACE, "emrdm_weights/train/sentinel/checkpoints/last.ckpt")
EMRDM = os.path.join(WORKSPACE, "EMRDM")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run(cmd, cwd=None):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

# ============ STEP 1: Install deps ============
print("=== Installing dependencies ===")
run("pip install pytorch-lightning==2.3.0 omegaconf einops --user")
run("pip install wandb natsort dctorch rasterio tifffile scipy opencv-python lpips s2cloudless --user")
run("pip install natten==0.21.1+torch280cu128 -f https://whl.natten.org --user")

# ============ STEP 2: Clone repos ============
print("=== Cloning repos ===")
os.makedirs(WORKSPACE, exist_ok=True)
if not os.path.exists(os.path.join(WORKSPACE, "cs159")):
    run(f"git clone https://github.com/weikaylee/cs159.git", cwd=WORKSPACE)
if not os.path.exists(EMRDM):
    run(f"git clone https://github.com/Ly403/EMRDM.git", cwd=WORKSPACE)

# ============ STEP 3: Download data ============
print("=== Downloading data ===")
run("python download_local_data.py", cwd=os.path.join(WORKSPACE, "cs159"))

# ============ STEP 4: Download weights ============
print("=== Downloading weights ===")
os.makedirs(os.path.dirname(CKPT), exist_ok=True)
if not os.path.exists(CKPT):
    run("pip install gdown --user")
    run(f'gdown "15iWa6CsJjy9RG2EWQKh1xPINxk8xYqBs" -O {CKPT}')

# ============ STEP 5: Fix pkl files ============
print("=== Fixing pkl files ===")
s2_files = glob.glob(f"{DATA_ROOT}/ROIs1158_spring_s2/**/*.tif", recursive=True)

triplets = []
for s2_path in s2_files:
    basename = os.path.basename(s2_path)
    suffix = basename.replace("ROIs1158_spring_s2_", "").replace(".tif", "")
    scene = suffix.split("_p")[0]
    s1_rel  = f"ROIs1158_spring_s1/s1_{scene}/ROIs1158_spring_s1_{suffix}.tif"
    s2_rel  = f"ROIs1158_spring_s2/s2_{scene}/ROIs1158_spring_s2_{suffix}.tif"
    s2c_rel = f"ROIs1158_spring_s2_cloudy/s2_cloudy_{scene}/ROIs1158_spring_s2_cloudy_{suffix}.tif"
    if os.path.exists(os.path.join(DATA_ROOT, s1_rel)) and os.path.exists(os.path.join(DATA_ROOT, s2c_rel)):
        triplets.append({"S1": s1_rel, "S2": s2_rel, "S2_cloudy": s2c_rel})

n = len(triplets)
train = triplets[:int(0.7*n)]
val   = triplets[int(0.7*n):int(0.85*n)]
test  = triplets[int(0.85*n):]

for split, data in [("train", train), ("val", val), ("test", test)]:
    with open(f"{DATA_ROOT}/all_{split}_paths.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"{split}: {len(data)} samples")

# ============ STEP 6: Edit yaml ============
print("=== Editing yaml ===")
with open(f"{EMRDM}/configs/example_training/sentinel.yaml") as f:
    content = f.read()

content = content.replace(
    '# ckpt_path: "" # your checkpoint path',
    f'ckpt_path: "{CKPT}"'
)
content = content.replace(
    'root: "/remote-home/share/dmb_nas2/liuyi/SEN12MSCR"',
    f'root: "{DATA_ROOT}"'
)
content = content.replace('devices: 2,4', 'devices: "1"')

yaml_path = f"{EMRDM}/configs/example_training/sentinel_dtu.yaml"
with open(yaml_path, "w") as f:
    f.write(content)
print("yaml written")

# ============ STEP 7: Patch natten API for 0.21.x ============
print("=== Patching natten API ===")
transformer_path = f"{EMRDM}/sgm/modules/diffusionmodules/k_diffusion/image_transformer.py"
with open(transformer_path) as f:
    code = f.read()
code = code.replace("if natten.has_fused_na():", "if False:  # natten API compat")
code = code.replace("if natten.has_gemm_na():", "if False:  # natten API compat")
code = code.replace("natten.use_fused_na(", "# natten.use_fused_na(")
code = code.replace("natten.use_gemm_na(", "# natten.use_gemm_na(")
code = code.replace("if False:  # natten API compat", "if True:  # natten 0.21.x layout")
with open(transformer_path, "w") as f:
    f.write(code)
print("patched")

# ============ STEP 8: Sanity check ============
print("=== Sanity check ===")
with open(f"{DATA_ROOT}/all_test_paths.pkl", "rb") as f:
    test_data = pickle.load(f)
print(f"Test samples: {len(test_data)}")
print(f"S2 exists: {os.path.exists(os.path.join(DATA_ROOT, test_data[0]['S2']))}")
print(f"Checkpoint exists: {os.path.exists(CKPT)}")

# ============ STEP 9: Run inference ============
print("=== Running inference ===")
os.chdir(EMRDM)
run(f"python main.py --base configs/example_training/sentinel_dtu.yaml --enable_tf32 -t false --no-test true --predict true")

# ============ STEP 10: Copy results ============
print("=== Copying results ===")
log_dirs = sorted(glob.glob(f"{EMRDM}/logs/*/sample"))
if log_dirs:
    latest = log_dirs[-1]
    run(f"cp -r {latest} {RESULTS_DIR}/")
    print(f"Results saved to {RESULTS_DIR}")
    for f in glob.glob(f"{latest}/*.png"):
        print(f"  {os.path.basename(f)}")