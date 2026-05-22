"""SEN12MSCRInterface — PyTorch Dataset for EMRDM inference on SEN12MS-CR.

Reads triplets from a pkl file produced by eval_emrdm.py (or diffusion_check.py)
and returns a batch dict that matches the keys expected by ResidualDiffusionEngine:

    "target"     (C, H, W) float32 [0, 1]  — cloud-free S2  (input_key)
    "S2"         (C, H, W) float32 [0, 1]  — cloudy S2       (mean_key)
    "S1S2"      (2+C, H, W) float32 [0,1]  — S1 ‖ S2_cloudy  (conditioner)
    "image_path"  str                        — basename of the S2_cloudy file

pkl format (relative paths from root):
    [{"S1": rel, "S2": rel, "S2_cloudy": rel}, ...]

Normalisation:
    S2  (uint16, [0, 10000] reflectance) ÷ 10000  →  [0, 1]
    S1  (float32, dB, approx [-25, 0])  linear clamp  →  [0, 1]
        # TODO: verify S1 dB range against your archive if results look wrong
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    import tifffile as _tifffile
    _HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
S2_SCALE   = 10_000.0
S1_DB_MIN  = -25.0
S1_DB_MAX  =   0.0


class SEN12MSCRInterface(Dataset):
    """Dataset interface for SEN12MS-CR used by DataModuleFromConfig.

    Args:
        root:         Path to the data directory containing pkl files and
                      ROIs* subdirectories.
        split:        One of "train", "val", "test".
        region:       Ignored (kept for yaml compatibility); use "all".
        rescale:      If True, normalise S2 to [0, 1] and S1 to [0, 1].
        cloud_masks:  Ignored (kept for yaml compatibility).
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        region: str = "all",
        rescale: bool = True,
        cloud_masks=None,
    ):
        self.root    = root
        self.rescale = rescale

        pkl_path = os.path.join(root, f"all_{split}_paths.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"pkl file not found: {pkl_path}\n"
                "Run eval_emrdm.py first to generate the split pkl files."
            )
        with open(pkl_path, "rb") as f:
            self.samples: List[Dict[str, str]] = pickle.load(f)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tif(path: str) -> np.ndarray:
        """Return (C, H, W) float32."""
        if _HAS_RASTERIO:
            with rasterio.open(path) as src:
                return src.read().astype(np.float32)
        else:
            data = _tifffile.imread(path).astype(np.float32)
            if data.ndim == 2:
                return data[np.newaxis]
            if data.ndim == 3 and data.shape[2] < data.shape[0]:
                return data.transpose(2, 0, 1)
            return data

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        s1  = self._read_tif(os.path.join(self.root, s["S1"]))     # (2,  H, W)
        s2  = self._read_tif(os.path.join(self.root, s["S2"]))     # (13, H, W)  cloud-free
        s2c = self._read_tif(os.path.join(self.root, s["S2_cloudy"]))  # (13, H, W)  cloudy

        if self.rescale:
            s2  = np.clip(s2  / S2_SCALE, 0.0, 1.0)
            s2c = np.clip(s2c / S2_SCALE, 0.0, 1.0)
            # TODO: verify S1_DB_MIN / S1_DB_MAX match your archive.
            s1  = np.clip((s1 - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN), 0.0, 1.0)

        # S1S2 = SAR (2ch) || cloudy S2 (13ch) = 15ch conditioning tensor
        s1s2 = np.concatenate([s1, s2c], axis=0)

        return {
            "target":     torch.from_numpy(s2).float(),    # cloud-free (what to predict)
            "S2":         torch.from_numpy(s2c).float(),   # cloudy     (diffusion mean)
            "S1S2":       torch.from_numpy(s1s2).float(),  # conditioning
            "image_path": os.path.basename(s["S2_cloudy"]),
        }
