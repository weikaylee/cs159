"""Evaluate EMRDM on SEN12MS-CR data from /data, write results to /result.

Workflow
--------
1. Scans data_root for all seasons (flat layout: ROIs{num}_{season}_s2/ etc.)
2. Builds and writes train/val/test pkl files that SEN12MSCRInterface expects
3. Patches image_transformer.py attention bug (matmul reshape fix)
4. Writes a modified sentinel_eval.yaml pointing to the correct paths
5. Runs `python main.py --predict true` as a subprocess
6. Collects generated TIF/PNG outputs from logs/ into result_dir

Usage
-----
    cd phase2_emrdm
    python eval_emrdm.py \\
        --data_root /resnick/groups/perona/oywang/cs159/data \\
        --checkpoint /path/to/last.ckpt \\
        --result_dir /result \\
        --split test \\
        --devices 1

New dependencies (install if missing)
--------------------------------------
    pip install natsort dctorch s2cloudless lpips omegaconf einops
    pip install pytorch-lightning==2.3.0
    # natten — must match your torch+cuda build, e.g.:
    pip install natten==0.17.1+torch220cu121 \\
        -f https://shi-labs.com/natten/wheels
"""

import argparse
import glob
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: build triplet index
# ---------------------------------------------------------------------------

def build_triplets(data_root: str, seasons=None) -> list:
    """Scan data_root for matched (S1, S2-clean, S2-cloudy) triplets.

    Handles the flat layout where ROIs* folders sit directly under data_root
    with variable ROI numbers per season (1158=spring, 1868=summer, etc.).

    Returns a list of dicts with relative paths (relative to data_root):
        {"S1": rel, "S2": rel, "S2_cloudy": rel}
    """
    roi_re = re.compile(r'^ROIs(\d+)_(\w+)_s2$')
    triplets = []

    for d in sorted(os.listdir(data_root)):
        m = roi_re.match(d)
        if not m:
            continue
        roi_num, season = m.group(1), m.group(2)
        if seasons and season not in seasons:
            continue

        clean_root  = os.path.join(data_root, f"ROIs{roi_num}_{season}_s2")
        cloudy_root = os.path.join(data_root, f"ROIs{roi_num}_{season}_s2_cloudy")
        sar_root    = os.path.join(data_root, f"ROIs{roi_num}_{season}_s1")

        if not all(os.path.isdir(p) for p in [clean_root, cloudy_root, sar_root]):
            continue

        # Index clean files by (scene_id, patch_id)
        clean_idx = _index_dir(clean_root,  roi_num, season, "s2")
        cloudy_idx = _index_dir(cloudy_root, roi_num, season, "s2_cloudy")
        sar_idx    = _index_dir(sar_root,    roi_num, season, "s1")

        for key in clean_idx:
            if key in cloudy_idx and key in sar_idx:
                triplets.append({
                    # Relative paths from data_root
                    "S2":       os.path.relpath(clean_idx[key],  data_root),
                    "S2_cloudy": os.path.relpath(cloudy_idx[key], data_root),
                    "S1":       os.path.relpath(sar_idx[key],    data_root),
                })

    return triplets


def _index_dir(root: str, roi_num: str, season: str, modality: str) -> dict:
    """Return {(scene_id, patch_id): abs_path} for all .tif under root."""
    re_pat = re.compile(
        rf"ROIs{roi_num}_{re.escape(season)}_{re.escape(modality)}_(\d+)_p(\d+)\.tif$"
    )
    index = {}
    for path in glob.glob(os.path.join(root, "**", "*.tif"), recursive=True):
        m = re_pat.search(os.path.basename(path))
        if m:
            index[(m.group(1), m.group(2))] = path
    return index


# ---------------------------------------------------------------------------
# Step 2: write pkl files
# ---------------------------------------------------------------------------

def write_pkl_files(
    data_root: str,
    triplets: list,
    val_fraction: float = 0.10,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split triplets and write all_{split}_paths.pkl to data_root.

    Returns {"train": [...], "val": [...], "test": [...]}
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(triplets)).tolist()

    n = len(idx)
    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))

    splits = {
        "train": [triplets[i] for i in idx[:n - n_val - n_test]],
        "val":   [triplets[i] for i in idx[n - n_val - n_test:n - n_test]],
        "test":  [triplets[i] for i in idx[n - n_test:]],
    }

    for split, data in splits.items():
        pkl_path = os.path.join(data_root, f"all_{split}_paths.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Wrote {split}: {len(data)} triplets → {pkl_path}")

    return splits


# ---------------------------------------------------------------------------
# Step 3: patch image_transformer.py
# ---------------------------------------------------------------------------

def patch_image_transformer(emrdm_dir: str) -> None:
    """Apply the attention matmul reshape fix (batch × heads must be merged)."""
    path = os.path.join(
        emrdm_dir,
        "sgm", "modules", "diffusionmodules",
        "k_diffusion", "image_transformer.py",
    )
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping patch")
        return

    with open(path) as f:
        code = f.read()

    old1 = (
        "B,H,X,Y,D = q.shape; "
        "_qk = torch.matmul(q.reshape(B,H,X*Y,D), k.reshape(B,H,X*Y,D).transpose(-2,-1)); "
        "_a = torch.softmax(_qk, dim=-1).to(v.dtype); "
        "x = torch.matmul(_a, v.reshape(B,H,X*Y,D)).reshape(B,H,X,Y,D)"
    )
    new1 = (
        "B,H,X,Y,D = q.shape; "
        "_qk = torch.matmul(q.reshape(B*H,X*Y,D), k.reshape(B*H,X*Y,D).transpose(-2,-1)); "
        "_a = torch.softmax(_qk, dim=-1).to(v.dtype); "
        "x = torch.matmul(_a, v.reshape(B*H,X*Y,D)).reshape(B,H,X,Y,D)"
    )

    old2 = (
        "B,H,X,Y,_ = qk.shape; "
        "x = torch.matmul(a, v.reshape(B,H,X*Y,v.shape[-1])).reshape(B,H,X,Y,v.shape[-1])"
    )
    new2 = (
        "B,H,X,Y,D = q.shape; "
        "x = torch.matmul(a.reshape(B*H,X*Y,X*Y), v.reshape(B*H,X*Y,D)).reshape(B,H,X,Y,D)"
    )

    patched = code.replace(old1, new1).replace(old2, new2)
    if patched == code:
        print("  image_transformer.py: already patched or pattern not found")
    else:
        with open(path, "w") as f:
            f.write(patched)
        print(f"  Patched {path}")


# ---------------------------------------------------------------------------
# Step 4: write modified YAML config
# ---------------------------------------------------------------------------

def write_eval_yaml(
    emrdm_dir: str,
    data_root: str,
    checkpoint: str,
    devices: str = "1",
) -> str:
    """Load sentinel.yaml, patch paths, write sentinel_eval.yaml. Returns path."""
    src = os.path.join(emrdm_dir, "configs", "example_training", "sentinel.yaml")
    dst = os.path.join(emrdm_dir, "configs", "example_training", "sentinel_eval.yaml")

    with open(src) as f:
        content = f.read()

    # Checkpoint
    content = content.replace(
        '# ckpt_path: "" # your checkpoint path',
        f'ckpt_path: "{checkpoint}"',
    )
    # In case it was already uncommented with a different path
    content = re.sub(
        r'^(\s*ckpt_path:\s*)".+"',
        f'\\g<1>"{checkpoint}"',
        content,
        flags=re.MULTILINE,
    )

    # Data root (all four split sections)
    content = re.sub(
        r'(root:\s*)"[^"]+"',
        f'\\1"{data_root}"',
        content,
    )

    # Devices
    content = re.sub(
        r'(devices:\s*)[\w,]+',
        f'\\1"{devices}"',
        content,
    )

    with open(dst, "w") as f:
        f.write(content)

    print(f"  Wrote eval config → {dst}")
    return dst


# ---------------------------------------------------------------------------
# Step 5: run inference
# ---------------------------------------------------------------------------

def run_inference(emrdm_dir: str, config_path: str) -> str:
    """Run EMRDM prediction and return the logs directory."""
    env = os.environ.copy()
    # Prepend emrdm_dir so the local sgm package is importable.
    # Use PYTHONPATH only if it was already set to avoid breaking pkg_resources
    # in environments where setuptools namespace packages rely on an empty path.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (emrdm_dir + os.pathsep + existing) if existing else emrdm_dir

    cmd = [
        sys.executable, "main.py",
        "--base", config_path,
        "--enable_tf32",
        "-t", "false",
        "--no-test", "true",
        "--predict", "true",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=emrdm_dir, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"main.py exited with code {result.returncode}")

    # Find the most recently created logs subdirectory
    log_dirs = sorted(
        glob.glob(os.path.join(emrdm_dir, "logs", "*")),
        key=os.path.getmtime,
    )
    if not log_dirs:
        raise RuntimeError("No logs directory found after inference")
    return log_dirs[-1]


# ---------------------------------------------------------------------------
# Step 6: collect results
# ---------------------------------------------------------------------------

def collect_results(log_dir: str, result_dir: str, split: str) -> int:
    """Copy generated outputs from log_dir/sample/ to result_dir.

    Organises into:
        result_dir/predicted/    ← model output (no suffix or _pred suffix)
        result_dir/input_cloudy/ ← mu / cloudy input
        result_dir/ground_truth/ ← target / cloud-free reference
    """
    sample_dir = os.path.join(log_dir, "sample")
    if not os.path.isdir(sample_dir):
        # Some EMRDM versions write directly to log_dir
        sample_dir = log_dir

    pred_dir  = os.path.join(result_dir, "predicted")
    input_dir = os.path.join(result_dir, "input_cloudy")
    gt_dir    = os.path.join(result_dir, "ground_truth")
    for d in [pred_dir, input_dir, gt_dir]:
        os.makedirs(d, exist_ok=True)

    n_copied = 0
    for src in sorted(glob.glob(os.path.join(sample_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(src):
            continue
        name = os.path.basename(src)

        if name.endswith("_mu.png") or name.endswith("_mu.tif"):
            dst = os.path.join(input_dir, name)
        elif name.endswith("_target.png") or name.endswith("_target.tif"):
            dst = os.path.join(gt_dir, name)
        elif name.endswith(".png") or name.endswith(".tif"):
            # prediction (no special suffix)
            dst = os.path.join(pred_dir, name)
        else:
            continue

        shutil.copy2(src, dst)
        n_copied += 1

    return n_copied


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EMRDM on SEN12MS-CR and save results."
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Root directory containing ROIs* folders (e.g. /data).",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the EMRDM checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--result_dir", default="/result",
        help="Directory to write cloud-free outputs (default: /result).",
    )
    parser.add_argument(
        "--emrdm_dir", default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to phase2_emrdm/ (defaults to this script's directory).",
    )
    parser.add_argument(
        "--seasons", nargs="*", default=None,
        help="Seasons to include (e.g. spring summer). Default = all found.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to run inference on (default: test).",
    )
    parser.add_argument(
        "--devices", default="1",
        help="GPU device(s) for PyTorch Lightning (e.g. '1' or '0,1').",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.10,
        help="Fraction of data for validation split.",
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.15,
        help="Fraction of data for test split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EMRDM evaluation on SEN12MS-CR")
    print(f"  data_root   : {args.data_root}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  result_dir  : {args.result_dir}")
    print(f"  emrdm_dir   : {args.emrdm_dir}")
    print(f"  split       : {args.split}")
    print(f"  devices     : {args.devices}")
    print("=" * 60)

    # ── 1. Build triplets ───────────────────────────────────────────────────
    print("\n[1/6] Scanning data root for triplets ...")
    triplets = build_triplets(args.data_root, seasons=args.seasons)
    if not triplets:
        print("ERROR: no matched triplets found. Check --data_root.")
        sys.exit(1)
    print(f"  Found {len(triplets)} triplets across all seasons.")

    # ── 2. Write pkl files ──────────────────────────────────────────────────
    print("\n[2/6] Writing pkl split files ...")
    write_pkl_files(
        args.data_root, triplets,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    # ── 3. Patch image_transformer.py ──────────────────────────────────────
    print("\n[3/6] Patching image_transformer.py ...")
    patch_image_transformer(args.emrdm_dir)

    # ── 4. Write eval YAML ──────────────────────────────────────────────────
    print("\n[4/6] Writing sentinel_eval.yaml ...")
    config_path = write_eval_yaml(
        args.emrdm_dir,
        args.data_root,
        args.checkpoint,
        devices=args.devices,
    )

    # ── 5. Run inference ────────────────────────────────────────────────────
    print("\n[5/6] Running EMRDM inference ...")
    log_dir = run_inference(args.emrdm_dir, config_path)
    print(f"  Outputs in: {log_dir}")

    # ── 6. Collect results ──────────────────────────────────────────────────
    print("\n[6/6] Collecting results ...")
    os.makedirs(args.result_dir, exist_ok=True)
    n = collect_results(log_dir, args.result_dir, args.split)
    print(f"  Copied {n} files to {args.result_dir}")
    print(f"    predicted/    ← cloud-free model outputs")
    print(f"    input_cloudy/ ← cloudy inputs (mu)")
    print(f"    ground_truth/ ← cloud-free references")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
