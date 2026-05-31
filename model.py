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
CKPT_DIR = os.path.join(CS159_DIR, "emrdm_weights/train/sentinel/checkpoints")

DATA_ROOT = os.environ.get("EMRDM_DATA_ROOT", os.path.join(CS159_DIR, "data"))
CKPT = os.environ.get("EMRDM_CKPT", os.path.join(CKPT_DIR, "last.ckpt"))
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

# ============ STEP 5: Build pkl files and run per-season inference ============
SEASONS = [
    {"roi": "1158", "season": "spring"},
    {"roi": "1868", "season": "summer"},
    {"roi": "1970", "season": "fall"},
    {"roi": "2017", "season": "winter"},
]

def build_triplets(roi, season):
    prefix = f"ROIs{roi}_{season}"
    s2_dir = f"{prefix}_s2"
    triplets = []
    for s2_path in glob.glob(f"{DATA_ROOT}/{s2_dir}/**/*.tif", recursive=True):
        basename = os.path.basename(s2_path)
        suffix = basename.replace(f"{s2_dir}_", "").replace(".tif", "")
        scene = suffix.split("_p")[0]
        s1_rel  = f"{prefix}_s1/s1_{scene}/{prefix}_s1_{suffix}.tif"
        s2_rel  = f"{s2_dir}/s2_{scene}/{s2_dir}_{suffix}.tif"
        s2c_rel = f"{prefix}_s2_cloudy/s2_cloudy_{scene}/{prefix}_s2_cloudy_{suffix}.tif"
        if (os.path.exists(os.path.join(DATA_ROOT, s1_rel)) and
                os.path.exists(os.path.join(DATA_ROOT, s2c_rel))):
            triplets.append({"S1": s1_rel, "S2": s2_rel, "S2_cloudy": s2c_rel})
    return triplets

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

print(f"Checkpoint exists: {os.path.exists(CKPT)}")

# ============ STEP 8–10: Per-season inference ============
os.chdir(EMRDM)
for s in SEASONS:
    roi, season = s["roi"], s["season"]
    s2_dir = f"ROIs{roi}_{season}_s2"
    print(f"\n=== Season: {season} ({s2_dir}) ===")

    triplets = build_triplets(roi, season)
    if not triplets:
        print(f"  No triplets found for {season}, skipping.")
        continue
    print(f"  {len(triplets)} triplets found")

    n = len(triplets)
    train = triplets[:int(0.7 * n)]
    val   = triplets[int(0.7 * n):int(0.85 * n)]
    test  = triplets[int(0.85 * n):]

    for split, data in [("train", train), ("val", val), ("test", test), ("predict", triplets)]:
        with open(f"{DATA_ROOT}/all_{split}_paths.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"  {split}: {len(data)}")

    print(f"  S2 exists: {os.path.exists(os.path.join(DATA_ROOT, triplets[0]['S2']))}")

    run("python main.py --base configs/example_training/sentinel_dtu.yaml --enable_tf32 -t false --no-test true --predict true")

    season_results = os.path.join(RESULTS_DIR, s2_dir)
    png_dir = os.path.join(season_results, "png")
    tif_dir = os.path.join(season_results, "tif")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)
    log_dirs = sorted(glob.glob(f"{EMRDM}/logs/*/sample"))
    if log_dirs:
        latest = log_dirs[-1]
        png_files = sorted(glob.glob(f"{latest}/*.png"))
        tif_files = sorted(glob.glob(f"{latest}/*.tif"))
        if png_files:
            run(f"cp {latest}/*.png {png_dir}/")
            print(f"  {len(png_files)} PNGs saved to {png_dir}")
        if tif_files:
            run(f"cp {latest}/*.tif {tif_dir}/")
            print(f"  {len(tif_files)} TIFs saved to {tif_dir}")

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