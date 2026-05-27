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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CS159_DIR = SCRIPT_DIR

DATA_ROOT = os.environ.get("EMRDM_DATA_ROOT", os.path.join(CS159_DIR, "data"))
CKPT = os.environ.get("EMRDM_CKPT", os.path.join(CS159_DIR, "last.ckpt"))
EMRDM = os.environ.get("EMRDM_REPO", os.path.join(CS159_DIR, "EMRDM"))
RESULTS_DIR = os.environ.get("EMRDM_RESULTS_DIR", os.path.join(CS159_DIR, "results"))
DEVICES = os.environ.get("EMRDM_DEVICES", "1")
INSTALL_DEPS = os.environ.get("EMRDM_INSTALL_DEPS", "0") == "1"
DOWNLOAD_DATA = os.environ.get("EMRDM_DOWNLOAD_DATA", "0") == "1"
DOWNLOAD_WEIGHTS = os.environ.get("EMRDM_DOWNLOAD_WEIGHTS", "0") == "1"
PATCH_NATTEN = os.environ.get("EMRDM_PATCH_NATTEN", "1") == "1"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run(cmd, cwd=None):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

# ============ STEP 1: Install deps (optional) ============
print("=== Installing dependencies (optional) ===")
if INSTALL_DEPS:
    run("pip install pytorch-lightning==2.3.0 omegaconf einops --user")
    run("pip install wandb natsort dctorch rasterio tifffile scipy opencv-python lpips s2cloudless --user")
else:
    print("Skipping dependency installation (set EMRDM_INSTALL_DEPS=1 to enable)")

# ============ STEP 2: Clone EMRDM (only if missing) ============
print("=== Ensuring EMRDM repo ===")
os.makedirs(CS159_DIR, exist_ok=True)
if not os.path.exists(EMRDM):
    run("git clone https://github.com/Ly403/EMRDM.git", cwd=CS159_DIR)

# ============ STEP 3: Download data (optional) ============
print("=== Downloading data (optional) ===")
if DOWNLOAD_DATA:
    run("python download_local_data.py", cwd=CS159_DIR)
else:
    print("Skipping data download (set EMRDM_DOWNLOAD_DATA=1 to enable)")

# ============ STEP 4: Download weights (optional) ============
print("=== Checking checkpoint ===")
if not os.path.exists(CKPT):
    if DOWNLOAD_WEIGHTS:
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        run("pip install gdown --user")
        run(f'gdown "15iWa6CsJjy9RG2EWQKh1xPINxk8xYqBs" -O {CKPT}')
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT}. Place it there or set EMRDM_DOWNLOAD_WEIGHTS=1."
        )

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
content = content.replace('devices: 2,4', f'devices: "{DEVICES}"')

yaml_path = f"{EMRDM}/configs/example_training/sentinel_dtu.yaml"
with open(yaml_path, "w") as f:
    f.write(content)
print("yaml written")

# ============ STEP 7: Patch natten API for 0.21.x (optional) ============
print("=== Patching natten API (optional) ===")
# Disabled: automatic patching interferes with hand-patched fallback code
print("Skipping automatic natten patch (set EMRDM_PATCH_NATTEN=1 to enable)")
if False and PATCH_NATTEN:
    transformer_path = f"{EMRDM}/sgm/modules/diffusionmodules/k_diffusion/image_transformer.py"
    if os.path.exists(transformer_path):
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
    else:
        print("Skipping natten patch (set EMRDM_PATCH_NATTEN=1 to enable)")

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

# # ============ STEP 11: Post-process PNGs for visualization ============
# print("=== Post-processing PNGs ===")
# try:
#     from PIL import Image
#     import numpy as _np

#     sample_dir = os.path.join(RESULTS_DIR, "sample")
#     png_files = glob.glob(f"{sample_dir}/*.png")

#     for png_path in sorted(png_files):
#         fname = os.path.basename(png_path)
#         img = _np.array(Image.open(png_path)).astype(_np.float32)

#         if img.ndim == 2:
#             img = img[:, :, _np.newaxis]

#         # Per-channel 2–98th percentile stretch so the full dynamic range is used.
#         # This fixes both the "too dark" and "mostly white" issues without altering
#         # the relative spatial content.
#         out = _np.empty_like(img)
#         for c in range(img.shape[2]):
#             p2, p98 = _np.percentile(img[:, :, c], [2, 98])
#             if p98 > p2:
#                 out[:, :, c] = (img[:, :, c] - p2) / (p98 - p2) * 255.0
#             else:
#                 out[:, :, c] = img[:, :, c]
#         out = _np.clip(out, 0, 255).astype(_np.uint8)

#         if out.shape[2] == 1:
#             Image.fromarray(out[:, :, 0]).save(png_path)
#         else:
#             Image.fromarray(out).save(png_path)
#         print(f"  post-processed {fname}")

#     print("Post-processing done.")
# except Exception as e:
#     print(f"Warning: post-processing failed ({e}); raw PNGs kept.")