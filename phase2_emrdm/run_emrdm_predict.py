#!/usr/bin/env python3
"""
Run EMRDM prediction on SEN12MSCR data and copy outputs to --output_dir.

Prerequisites
-------------
1. sgm/data/ must be present (clone https://github.com/Ly403/EMRDM and copy
   sgm/data/ into this directory — it is not committed to this repo).
2. natten must be installed:
       pip install natten==0.21.6+torch2100cu128 -f https://whl.natten.org
3. A trained checkpoint (.ckpt) must exist.

Usage
-----
    cd phase2_emrdm
    python run_emrdm_predict.py \
        --data_root /resnick/groups/perona/oywang/cs159/data \
        --ckpt_path /resnick/groups/perona/oywang/cs159/emrdm_weights/train/sentinel/checkpoints \
        --output_dir /resnick/groups/perona/oywang/cs159/output
"""

import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_prerequisites():
    missing = []
    if not os.path.isdir(os.path.join(CODE_DIR, "sgm", "data")):
        missing.append(
            "sgm/data/ is missing. Clone https://github.com/Ly403/EMRDM "
            "and copy its sgm/data/ directory here."
        )
    try:
        import natten  # noqa: F401
    except ImportError:
        missing.append(
            "natten not installed. Run:\n"
            "  pip install natten==0.21.6+torch2100cu128 -f https://whl.natten.org"
        )
    if missing:
        print("ERROR: prerequisites not met:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


# All seasons present in SEN12MS-CR: (ROI-id, season-name)
SEASONS = [
    ("ROIs1158", "spring"),
    ("ROIs1868", "summer"),
    ("ROIs1970", "fall"),
    ("ROIs2017", "winter"),
]


def build_pkl_files(data_root):
    """Scan data_root for aligned S1/S2/S2_cloudy triplets across all seasons."""
    triplets = []
    found_seasons = []

    for roi, season in SEASONS:
        s2_dir = f"{roi}_{season}_s2"
        s2_files = glob.glob(f"{data_root}/{s2_dir}/**/*.tif", recursive=True)
        if not s2_files:
            continue
        found_seasons.append(f"{season}({len(s2_files)} S2 files)")
        prefix = f"{roi}_{season}_s2_"
        for s2_path in s2_files:
            basename = os.path.basename(s2_path)
            suffix = basename.replace(prefix, "").replace(".tif", "")
            scene = suffix.split("_p")[0]
            s1_rel  = f"{roi}_{season}_s1/s1_{scene}/{roi}_{season}_s1_{suffix}.tif"
            s2_rel  = f"{roi}_{season}_s2/s2_{scene}/{roi}_{season}_s2_{suffix}.tif"
            s2c_rel = f"{roi}_{season}_s2_cloudy/s2_cloudy_{scene}/{roi}_{season}_s2_cloudy_{suffix}.tif"
            if (os.path.exists(os.path.join(data_root, s1_rel)) and
                    os.path.exists(os.path.join(data_root, s2c_rel))):
                triplets.append({"S1": s1_rel, "S2": s2_rel, "S2_cloudy": s2c_rel})

    if not triplets:
        raise FileNotFoundError(
            f"No complete S1/S2/S2_cloudy triplets found in {data_root}. "
            "Expected directories like ROIs1158_spring_s2/, ROIs1158_spring_s1/, "
            "ROIs1158_spring_s2_cloudy/ (and/or other seasons). "
            "Run download_all_data.py first."
        )

    print(f"  Seasons found: {', '.join(found_seasons)}")
    print(f"  Total triplets: {len(triplets)}")

    n = len(triplets)
    splits = {
        "train": triplets[:int(0.7 * n)],
        "val":   triplets[int(0.7 * n):int(0.85 * n)],
        "test":  triplets[int(0.85 * n):],
    }
    for split, data in splits.items():
        out = os.path.join(data_root, f"all_{split}_paths.pkl")
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"  {split}: {len(data)} samples → {out}")
    return splits


def patch_natten():
    """
    Ensure image_transformer.py uses the natten API matching the installed version.

    natten 0.17.x (torch220cu121): uses (n, nh, h, w, e) layout — the default
      'if False' branch in the code; no patch needed.
    natten 0.21.x (torch2100cu128): uses (n, h, w, nh, e) layout — requires
      switching the branch to 'if True'.
    """
    from importlib.metadata import version as pkg_version
    natten_ver = tuple(int(x) for x in pkg_version("natten").split("+")[0].split(".")[:2])
    need_021_patch = natten_ver >= (0, 21)

    transformer_path = os.path.join(
        CODE_DIR, "sgm", "modules", "diffusionmodules",
        "k_diffusion", "image_transformer.py"
    )
    with open(transformer_path) as f:
        code = f.read()

    if need_021_patch:
        patched = code.replace(
            "if False:  # natten API compat",
            "if True:  # use natten 0.21.6 compatible layout",
        )
        verb = "applied (natten ≥ 0.21)"
    else:
        # Revert if someone previously applied the 0.21 patch
        patched = code.replace(
            "if True:  # use natten 0.21.6 compatible layout",
            "if False:  # natten API compat",
        )
        verb = "reverted to 0.17 layout" if patched != code else "not needed (natten 0.17 default)"

    if patched != code:
        with open(transformer_path, "w") as f:
            f.write(patched)
    print(f"  natten patch: {verb} (installed: {pkg_version('natten')})")


def write_hpc_config(data_root, ckpt_path):
    """Derive sentinel_hpc.yaml from sentinel.yaml with correct paths for this run."""
    src = os.path.join(CODE_DIR, "configs", "example_training", "sentinel.yaml")
    dst = os.path.join(CODE_DIR, "configs", "example_training", "sentinel_hpc.yaml")

    cfg = OmegaConf.load(src)

    cfg.model.params.ckpt_path = ckpt_path

    for split in ("train", "validation", "test", "predict"):
        cfg.data.params[split].params.root = data_root

    # Single GPU for prediction; trainer.devices=1 means "use 1 auto-selected GPU"
    cfg.lightning.trainer.devices = 1

    OmegaConf.save(cfg, dst)
    print(f"  Config written → {dst}")
    return dst


def run_prediction(config_path):
    cmd = [
        sys.executable, "main.py",
        "--base", config_path,
        "--enable_tf32",
        "-t", "false",
        "--no-test", "true",
        "--predict", "true",
    ]
    print("  Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=CODE_DIR)
    if result.returncode != 0:
        print(f"main.py exited with code {result.returncode}.")
        sys.exit(result.returncode)


def collect_outputs(output_dir):
    """
    Find the most recent logs/*/sample/ directory, copy all files to output_dir,
    and write a results_comparison.png grid (up to 20 patches, 3 columns:
    input mu | prediction | ground truth).
    """
    os.makedirs(output_dir, exist_ok=True)

    # predict_step saves to logger.save_dir + "/sample/"
    # CSVLogger.save_dir == logdir (e.g. logs/2026-05-25T12-00-00_sentinel_hpc)
    sample_dirs = sorted(glob.glob(os.path.join(CODE_DIR, "logs", "*", "sample")))
    if not sample_dirs:
        print("  No logs/*/sample/ directory found. Nothing to collect.")
        return
    sample_dir = sample_dirs[-1]
    print(f"  Collecting from: {sample_dir}")

    all_pngs = glob.glob(os.path.join(sample_dir, "*.png"))
    all_tifs = glob.glob(os.path.join(sample_dir, "*.tif"))

    for f in all_pngs + all_tifs:
        shutil.copy2(f, os.path.join(output_dir, os.path.basename(f)))
    print(f"  Copied {len(all_pngs)} PNGs + {len(all_tifs)} TIFs → {output_dir}")

    # Build comparison grid: rows=patches, cols=[mu, prediction, target]
    pred_pngs = [
        f for f in all_pngs
        if not os.path.basename(f).endswith("_mu.png")
        and not os.path.basename(f).endswith("_target.png")
    ]
    patches = sorted({os.path.splitext(os.path.basename(f))[0] for f in pred_pngs})[:20]
    if not patches:
        print("  No prediction PNGs found — skipping comparison grid.")
        return

    fig, axes = plt.subplots(len(patches), 3, figsize=(12, 4 * len(patches)))
    if len(patches) == 1:
        axes = [axes]

    cols = [("_mu.png", "Input (mu)"), (".png", "Prediction"), ("_target.png", "Ground Truth")]
    for i, patch in enumerate(patches):
        for j, (suffix, title) in enumerate(cols):
            candidates = glob.glob(os.path.join(sample_dir, f"{patch}{suffix}"))
            if candidates:
                axes[i][j].imshow(mpimg.imread(candidates[0]))
                axes[i][j].set_title(f"{patch} — {title}", fontsize=8)
            axes[i][j].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "results_comparison.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"  Saved comparison grid → {grid_path}")

    # Also copy per-patch metrics CSV if present
    metrics_csvs = glob.glob(os.path.join(os.path.dirname(sample_dir), "metrics.csv"))
    for f in metrics_csvs:
        shutil.copy2(f, os.path.join(output_dir, "predict_metrics.csv"))
        print(f"  Copied metrics → {output_dir}/predict_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description="Run EMRDM prediction on SEN12MSCR data.")
    parser.add_argument(
        "--data_root", required=True,
        help="Root dir containing ROIs1158_spring_s2/, ROIs1158_spring_s1/, ROIs1158_spring_s2_cloudy/",
    )
    parser.add_argument(
        "--ckpt_path", required=True,
        help="Path to trained EMRDM checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output_dir", default="/output",
        help="Where to copy generated images (default: /output)",
    )
    parser.add_argument(
        "--skip_pkl", action="store_true",
        help="Skip pkl rebuild if all_train/val/test_paths.pkl already exist",
    )
    args = parser.parse_args()

    print("=== Checking prerequisites ===")
    check_prerequisites()

    if not os.path.isfile(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    print("\n=== [1/4] Building triplet pkl files ===")
    if args.skip_pkl and all(
        os.path.exists(os.path.join(args.data_root, f"all_{s}_paths.pkl"))
        for s in ("train", "val", "test")
    ):
        print("  pkl files already exist — skipping (--skip_pkl).")
    else:
        build_pkl_files(args.data_root)

    print("\n=== [2/4] Patching natten API ===")
    patch_natten()

    print("\n=== [3/4] Writing HPC config ===")
    config_path = write_hpc_config(args.data_root, args.ckpt_path)

    print("\n=== [4/4] Running EMRDM prediction ===")
    run_prediction(config_path)

    print(f"\n=== Collecting outputs → {args.output_dir} ===")
    collect_outputs(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
