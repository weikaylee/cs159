"""Smoke test: Phase 1 mirror-map training runs end-to-end on dummy data.

What this test does
-------------------
1. Builds a tiny in-memory dataset of 13-band tensors shaped like SEN12MS-CR
   patches after dataset.py's /10000 normalisation.
2. Patches phase1_mirror_map.train_mirror_map.build_dataloaders to return
   loaders over that dummy dataset, plus sys.argv to feed minimal CLI args.
3. Invokes train_mirror_map.main() with --wandb enabled (online mode).
4. Asserts:
     - best.pt and last.pt exist (final.pt is no longer written)
     - history.csv has both train and val rows with all metric columns
     - the wandb run's on-disk summary contains all expected train/* and
       val/* metric keys

Requires
--------
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

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_mirror_map"))

import train_mirror_map  # noqa: E402


class _DummyS2Dataset(Dataset):
    """In-memory stand-in for SEN12MSCRCloudFreeDataset.

    Returns (13, 64, 64) float32 tensors in [0, 1] — the same shape and dtype
    real Sentinel-2 patches take after /10000 reflectance normalisation.
    """

    def __init__(self, n: int, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 13, 64, 64, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def _dummy_build_dataloaders(*args, **kwargs):
    train_loader = DataLoader(_DummyS2Dataset(4, seed=42), batch_size=2,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(_DummyS2Dataset(2, seed=43), batch_size=2,
                            num_workers=0)
    return train_loader, val_loader


def test_training_smoke():
    """Phase 1 training completes one epoch end-to-end on dummy data and
    streams metrics to W&B online."""
    with tempfile.TemporaryDirectory() as tmp:
        argv = [
            "train_mirror_map.py",
            "--data_root", tmp,
            "--output_dir", tmp,
            "--epochs", "1",
            "--batch_size", "2",
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
             patch.object(sys, "argv", argv), \
             patch.object(train_mirror_map, "build_dataloaders",
                          _dummy_build_dataloaders):
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
