#!/usr/bin/env python3
"""
Run EMRDM prediction on the full SEN12MSCR dataset and copy outputs to --output_dir.

Patches sentinel.yaml in-place with the checkpoint path, data root, and single-GPU
prediction settings. Builds a single all_predict_paths.pkl with every available
S1/S2/S2_cloudy triplet (no train/val/test split).

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
        --ckpt_path /resnick/groups/perona/oywang/cs159/emrdm_weights/train/sentinel/checkpoints/last.ckpt \
        --output_dir /resnick/groups/perona/oywang/cs159/output/emrdm_predict
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
SENTINEL_YAML = os.path.join(CODE_DIR, "configs", "example_training", "sentinel.yaml")

# All seasons present in SEN12MS-CR: (ROI-id, season-name)
SEASONS = [
    ("ROIs1158", "spring"),
    ("ROIs1868", "summer"),
    ("ROIs1970", "fall"),
    ("ROIs2017", "winter"),
]


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


def build_predict_pkl(data_root, max_samples=None):
    """Scan data_root for all aligned S1/S2/S2_cloudy triplets. No split."""
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

    if max_samples is not None:
        triplets = triplets[:max_samples]
        print(f"  Limiting to {len(triplets)} samples (--max_samples)")

    out = os.path.join(data_root, "all_predict_paths.pkl")
    with open(out, "wb") as f:
        pickle.dump(triplets, f)
    print(f"  {len(triplets)} triplets → {out}")
    return len(triplets)


def patch_natten():
    """
    Ensure image_transformer.py uses the natten API matching the installed version.

    natten 0.17.x (torch220cu121): uses (n, nh, h, w, e) layout — the default
      'if False' branch in the code; no patch needed.
    natten 0.21.x (torch2100cu128): uses (n, h, w, nh, e) layout — requires
      switching the branch to 'if True'.
    """
    from importlib.metadata import version as pkg_version, PackageNotFoundError
    try:
        natten_ver_str = pkg_version("natten").split("+")[0]
    except PackageNotFoundError:
        try:
            import natten as _natten
            natten_ver_str = _natten.__version__.split("+")[0]
        except ModuleNotFoundError:
            print("  natten patch: skipped (natten not installed on this node)")
            return
    natten_ver = tuple(int(x) for x in natten_ver_str.split(".")[:2])
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
        patched = code.replace(
            "if True:  # use natten 0.21.6 compatible layout",
            "if False:  # natten API compat",
        )
        verb = "reverted to 0.17 layout" if patched != code else "not needed (natten 0.17 default)"

    if patched != code:
        with open(transformer_path, "w") as f:
            f.write(patched)
    print(f"  natten patch: {verb} (installed: {pkg_version('natten')})")


def patch_sentinel_yaml(data_root, ckpt_path):
    """Patch sentinel.yaml in-place: checkpoint path, data root, single-GPU predict."""
    cfg = OmegaConf.load(SENTINEL_YAML)

    cfg.model.params.ckpt_path = ckpt_path

    # patch all split roots so PL doesn't look for pkl files at a stale path
    for split in ("train", "validation", "test", "predict"):
        if split in cfg.data.params:
            cfg.data.params[split].params.root = data_root

    cfg.data.params.predict.params.split = "predict"

    cfg.lightning.trainer.devices = 1

    # Override DDPStrategy (main.py's default) with single-device to avoid NCCL on 1 GPU
    cfg.lightning.strategy = OmegaConf.create({
        "target": "pytorch_lightning.strategies.SingleDeviceStrategy",
        "params": {"device": "cuda:0"},
    })

    OmegaConf.save(cfg, SENTINEL_YAML)
    print(f"  Patched in-place: {SENTINEL_YAML}")


def run_prediction():
    cmd = [
        sys.executable, "main.py",
        "--base", SENTINEL_YAML,
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


def _season_of(fname):
    for _, season in SEASONS:
        if f"_{season}_" in fname:
            return season
    return "unknown"


def _subdir_of(fname):
    if fname.endswith("_mu.png"):
        return "inputs"
    if fname.endswith("_target.png"):
        return "targets"
    return "predictions"


def collect_outputs(output_dir):
    """
    Copy predictions into output_dir/{season}/{predictions|targets|inputs}/ and
    write a per-season results_comparison.png grid (up to 20 patches, 3 columns).
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_dirs = sorted(glob.glob(os.path.join(CODE_DIR, "logs", "*", "sample")))
    if not sample_dirs:
        print("  No logs/*/sample/ directory found. Nothing to collect.")
        return
    sample_dir = sample_dirs[-1]
    print(f"  Collecting from: {sample_dir}")

    all_pngs = glob.glob(os.path.join(sample_dir, "*.png"))
    all_tifs = glob.glob(os.path.join(sample_dir, "*.tif"))

    counts = {}
    for f in all_pngs + all_tifs:
        fname = os.path.basename(f)
        season = _season_of(fname)
        subdir = _subdir_of(fname)
        dest = os.path.join(output_dir, season, subdir)
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(f, os.path.join(dest, fname))
        counts[(season, subdir)] = counts.get((season, subdir), 0) + 1

    for (season, subdir), n in sorted(counts.items()):
        print(f"  {season}/{subdir}: {n} files")

    # Per-season comparison grid
    for _, season in SEASONS:
        pred_dir = os.path.join(output_dir, season, "predictions")
        if not os.path.isdir(pred_dir):
            continue
        pred_pngs = glob.glob(os.path.join(pred_dir, "*.png"))
        patches = sorted({os.path.splitext(os.path.basename(f))[0] for f in pred_pngs})[:20]
        if not patches:
            continue

        _, axes = plt.subplots(len(patches), 3, figsize=(12, 4 * len(patches)))
        if len(patches) == 1:
            axes = [axes]

        cols = [
            ("inputs",      "_mu.png",     "Input (mu)"),
            ("predictions", ".png",        "Prediction"),
            ("targets",     "_target.png", "Ground Truth"),
        ]
        for i, patch in enumerate(patches):
            for j, (subdir, suffix, title) in enumerate(cols):
                path = os.path.join(output_dir, season, subdir, patch + suffix)
                if os.path.exists(path):
                    axes[i][j].imshow(mpimg.imread(path))
                    axes[i][j].set_title(f"{patch} — {title}", fontsize=8)
                axes[i][j].axis("off")

        plt.tight_layout()
        grid_path = os.path.join(output_dir, season, "results_comparison.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"  Saved comparison grid → {grid_path}")

    metrics_csvs = glob.glob(os.path.join(os.path.dirname(sample_dir), "metrics.csv"))
    for f in metrics_csvs:
        shutil.copy2(f, os.path.join(output_dir, "predict_metrics.csv"))
        print(f"  Copied metrics → {output_dir}/predict_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description="Run EMRDM prediction on full SEN12MSCR dataset.")
    parser.add_argument(
        "--data_root", required=True,
        help="Root dir containing ROIs*_s2/, ROIs*_s1/, ROIs*_s2_cloudy/ season directories",
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
        help="Skip pkl rebuild if all_predict_paths.pkl already exists",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit prediction to first N samples (useful for test runs)",
    )
    args = parser.parse_args()

    print("=== Checking prerequisites ===")
    check_prerequisites()

    if not os.path.isfile(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    print("\n=== [1/4] Building predict pkl (full dataset) ===")
    pkl_path = os.path.join(args.data_root, "all_predict_paths.pkl")
    if args.skip_pkl and os.path.exists(pkl_path):
        print(f"  {pkl_path} already exists — skipping (--skip_pkl).")
    else:
        build_predict_pkl(args.data_root, max_samples=args.max_samples)

    print("\n=== [2/4] Patching natten API ===")
    patch_natten()

    print("\n=== [3/4] Patching sentinel.yaml ===")
    patch_sentinel_yaml(args.data_root, args.ckpt_path)

    print("\n=== [4/4] Running EMRDM prediction ===")
    run_prediction()

    print(f"\n=== Collecting outputs → {args.output_dir} ===")
    collect_outputs(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
