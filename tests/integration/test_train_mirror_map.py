"""Smoke test: Phase 1 mirror-map training runs end-to-end on real data.

What this test does
-------------------
1. Locates the local SEN12MS-CR subset under <repo>/data/. Skips if absent
   (run `bash download_local_data.sh` from the repo root to fetch it).
2. Invokes train_mirror_map.main() pointed at that data, with --patch_size
   64 so the real 256x256 patches are random-cropped down for a fast run,
   and --wandb enabled (online mode).
3. Asserts:
     - best.pt and last.pt exist (final.pt is no longer written)
     - history.csv has both train and val rows with all metric columns
     - the wandb run's on-disk summary contains all expected train/* and
       val/* metric keys

Requires
--------
- A local data subset at <repo>/data/ROIs1158_spring_s2_cloudfree/...
  (fetch via `bash download_local_data.sh`).
- `wandb login` previously run in terminal (credentials in ~/.netrc).
- Network access to api.wandb.ai.
- VGG16 weights cached under ~/.cache/torch (downloaded on first run,
  ~528 MB).

WANDB_DIR is pointed at the tempdir so the repo's working tree stays
clean. The run is tagged with run name "test_train_mirror_map" in the
shared `cs159` wandb project — distinguish real runs from test runs by
name.

Run:
    pytest tests/integration/test_train_mirror_map.py -v
    # or, without pytest:
    python tests/integration/test_train_mirror_map.py
"""

import csv
import glob
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_mirror_map"))

import train_mirror_map  # noqa: E402

DATA_ROOT = os.path.join(REPO_ROOT, "data")
S2_CLOUDFREE_DIR = os.path.join(DATA_ROOT, "ROIs1158_spring_s2")


def test_training_smoke():
    """Phase 1 training completes one epoch end-to-end on real local data
    and streams metrics to W&B online."""
    if not os.path.isdir(S2_CLOUDFREE_DIR):
        pytest.skip(
            f"local data subset not found at {S2_CLOUDFREE_DIR} — "
            f"run `bash download_local_data.sh` from the repo root"
        )

    with tempfile.TemporaryDirectory() as tmp:
        argv = [
            "train_mirror_map.py",
            "--data_root", DATA_ROOT,
            "--output_dir", tmp,
            "--epochs", "1",
            "--batch_size", "2",
            "--patch_size", "64",
            "--ngf", "8",
            "--n_res_blocks", "1",
            "--icnn_filters", "8",
            "--icnn_layers", "2",
            "--num_workers", "0",
            "--log_every", "1",
            "--wandb",
            "--wandb_project", "tests",
            "--wandb_run_name", "train_mirror_map",
            "--wandb_entity", "cs159",
        ]
        with patch.dict(os.environ, {"WANDB_DIR": tmp}), \
             patch.object(sys, "argv", argv):
            train_mirror_map.main()

        # Checkpoints: best.pt + last.pt only (final.pt + ckpt_ep* gone)
        assert os.path.exists(os.path.join(tmp, "best.pt")), "best.pt missing"
        assert os.path.exists(os.path.join(tmp, "last.pt")), "last.pt missing"
        assert not os.path.exists(os.path.join(tmp, "final.pt")), \
            "final.pt should no longer be written"

        # history.csv — must have train + val rows, all metric columns
        history = os.path.join(tmp, "history.csv")
        assert os.path.exists(history), "history.csv missing"
        with open(history) as f:
            rows = list(csv.DictReader(f))
        assert any(r["phase"] == "train" for r in rows), "no train rows in history.csv"
        assert any(r["phase"] == "val" for r in rows),   "no val rows in history.csv"
        metric_cols = {"loss", "l_cycle", "l_constr", "l_reg",
                       "mae", "sam", "psnr", "ssim"}
        for row in rows:
            missing_cols = metric_cols - set(row)
            assert not missing_cols, \
                f"missing history.csv cols: {missing_cols} in row {row}"

        # Wandb summary — every metric key appears
        run_dirs = glob.glob(os.path.join(tmp, "wandb", "run-*"))
        assert run_dirs, "no wandb run dir was created under WANDB_DIR"
        summary_path = os.path.join(run_dirs[0], "files", "wandb-summary.json")
        assert os.path.exists(summary_path), \
            f"wandb-summary.json not written at {summary_path}"
        with open(summary_path) as f:
            summary = json.load(f)
        expected_wandb_keys = {
            "train/loss", "train/l_cycle", "train/l_constr", "train/l_reg",
            "train/mae", "train/sam", "train/psnr", "train/ssim",
            "train/lr", "train/epoch",
            "val/loss", "val/l_cycle", "val/l_constr", "val/l_reg",
            "val/mae", "val/sam", "val/psnr", "val/ssim",
            "val/best_loss",
        }
        missing = expected_wandb_keys - set(summary)
        assert not missing, f"missing wandb metrics in summary: {missing}"


if __name__ == "__main__":
    test_training_smoke()
    print("PASS: train_mirror_map smoke test")
