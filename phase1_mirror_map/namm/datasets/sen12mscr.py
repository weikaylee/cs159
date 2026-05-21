"""SEN12MS-CR dataset loader for NAMM training.

Reads aligned triplets (cloudy S2, cloud-free S2, S1 SAR) directly from
.tif files; no TFRecord pre-conversion required.

Value ranges assumed:
    S2 (cloudy and cloud-free): uint16, [0, 10000] reflectance ÷ 10000 → [0.0, 1.0]
    S1: float32, dB values in approximately [-25, 0] → linearly mapped to [0.0, 1.0]
        # TODO: verify S1 dB min/max on your specific dataset; defaults are S1_DB_MIN=-25, S1_DB_MAX=0

Returns per sample:
    "cloudy": (C, H, W) float32 tensor in [0, 1]  — cloudy Sentinel-2
    "clean":  (C, H, W) float32 tensor in [0, 1]  — cloud-free Sentinel-2
    "sar":    (2, H, W) float32 tensor in [0, 1]  — Sentinel-1 VV+VH
"""

import glob
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    import tifffile as _tifffile
    _HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

S2_SCALE: float = 10_000.0

# TODO: confirm S1 dB range on the SEN12MS-CR archive you are using.
S1_DB_MIN: float = -25.0
S1_DB_MAX: float = 0.0


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for SEN12MS-CR cloud removal.

    Expected directory layout under *data_root*::

        {data_root}/
        └── {season}/
            ├── ROIs1158_{season}_s2/
            │   └── s2_{id}/
            │       └── ROIs1158_{season}_s2_{id}_p{patch}.tif
            ├── ROIs1158_{season}_s2_cloudy/
            │   └── s2_{id}/
            │       └── ROIs1158_{season}_s2_cloudy_{id}_p{patch}.tif
            └── ROIs1158_{season}_s1/
                └── s1_{id}/
                    └── ROIs1158_{season}_s1_{id}_p{patch}.tif

    Each .tif is a 256×256 multi-band GeoTIFF. Triplets are matched by
    (season, scene-id, patch-index).
    """

    def __init__(
        self,
        data_root: str,
        seasons: Optional[List[str]] = None,
        num_channels: int = 13,
        patch_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_root:    Path to the root data directory (e.g. /data/).
            seasons:      Seasons to include; None scans all sub-directories.
            num_channels: Number of S2 bands to keep (leading bands). 13 = all;
                          4 = B2/B3/B4/B8 subset used by the default config.
                          # TODO: confirm band ordering in your tif files.
            patch_size:   If set, take a random crop of this spatial size.
                          Set to 64 to match the default NAMM config.
            seed:         RNG seed for random crops.
        """
        self.data_root = data_root
        self.num_channels = num_channels
        self.patch_size = patch_size
        self._rng = np.random.RandomState(seed)

        self.triplets: List[Dict[str, str]] = self._build_index(seasons)
        if not self.triplets:
            raise RuntimeError(
                f"No matched triplets found under {data_root!r}. "
                "Check that the directory layout matches the expected pattern."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self, seasons: Optional[List[str]]) -> List[Dict[str, str]]:
        if seasons is None:
            try:
                seasons = [
                    d for d in os.listdir(self.data_root)
                    if os.path.isdir(os.path.join(self.data_root, d))
                ]
            except FileNotFoundError:
                return []

        triplets: List[Dict[str, str]] = []
        for season in seasons:
            season_dir = os.path.join(self.data_root, season)
            clean_root  = os.path.join(season_dir, f"ROIs1158_{season}_s2")
            cloudy_root = os.path.join(season_dir, f"ROIs1158_{season}_s2_cloudy")
            sar_root    = os.path.join(season_dir, f"ROIs1158_{season}_s1")

            if not all(os.path.isdir(p) for p in [clean_root, cloudy_root, sar_root]):
                continue

            clean_idx  = self._index_dir(clean_root,  season, "s2")
            cloudy_idx = self._index_dir(cloudy_root, season, "s2_cloudy")
            sar_idx    = self._index_dir(sar_root,    season, "s1")

            for key in clean_idx:
                if key in cloudy_idx and key in sar_idx:
                    triplets.append({
                        "clean":  clean_idx[key],
                        "cloudy": cloudy_idx[key],
                        "sar":    sar_idx[key],
                    })

        return triplets

    @staticmethod
    def _index_dir(
        root: str, season: str, modality: str
    ) -> Dict[Tuple[str, str], str]:
        """Return {(scene_id, patch_id): path} for all .tif files under *root*."""
        pattern = os.path.join(root, "**", "*.tif")
        index: Dict[Tuple[str, str], str] = {}
        # Filename pattern: ROIs1158_{season}_{modality}_{id}_p{patch}.tif
        re_pat = re.compile(
            rf"ROIs1158_{re.escape(season)}_{re.escape(modality)}_(\d+)_p(\d+)\.tif$"
        )
        for path in glob.glob(pattern, recursive=True):
            m = re_pat.search(os.path.basename(path))
            if m:
                index[(m.group(1), m.group(2))] = path
        return index

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tif(path: str) -> np.ndarray:
        """Read a GeoTIFF and return a (C, H, W) float32 array."""
        if _HAS_RASTERIO:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)  # (C, H, W)
        else:
            data = _tifffile.imread(path).astype(np.float32)
            if data.ndim == 2:
                data = data[np.newaxis]           # (1, H, W)
            elif data.ndim == 3 and data.shape[2] <= data.shape[0]:
                data = data.transpose(2, 0, 1)   # (H, W, C) → (C, H, W)
        return data

    @staticmethod
    def _norm_s2(data: np.ndarray) -> np.ndarray:
        return np.clip(data / S2_SCALE, 0.0, 1.0)

    @staticmethod
    def _norm_s1(data: np.ndarray) -> np.ndarray:
        # TODO: confirm S1_DB_MIN / S1_DB_MAX match your archive.
        return np.clip(
            (data - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN), 0.0, 1.0
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.triplets[idx]

        clean  = self._norm_s2(self._read_tif(t["clean"]))   # (13, H, W)
        cloudy = self._norm_s2(self._read_tif(t["cloudy"]))  # (13, H, W)
        sar    = self._norm_s1(self._read_tif(t["sar"]))     # (2,  H, W)

        # Keep only the requested number of S2 bands (leading channels).
        # TODO: if your files use a different band ordering, adjust here.
        if self.num_channels < clean.shape[0]:
            clean  = clean[:self.num_channels]
            cloudy = cloudy[:self.num_channels]

        # Optional random spatial crop.
        if self.patch_size is not None:
            _, H, W = clean.shape
            if H < self.patch_size or W < self.patch_size:
                raise ValueError(
                    f"patch_size={self.patch_size} exceeds image size ({H}×{W})."
                )
            top  = self._rng.randint(0, H - self.patch_size + 1)
            left = self._rng.randint(0, W - self.patch_size + 1)
            sl = (slice(top, top + self.patch_size),
                  slice(left, left + self.patch_size))
            clean  = clean[:, sl[0], sl[1]]
            cloudy = cloudy[:, sl[0], sl[1]]
            sar    = sar[:,   sl[0], sl[1]]

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clean":  torch.from_numpy(clean),
            "sar":    torch.from_numpy(sar),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    seasons: Optional[List[str]] = None,
    num_channels: int = 13,
    patch_size: Optional[int] = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for the SEN12MS-CR dataset.

    Args:
        data_root:     Path to the root data directory (e.g. /data/).
        split:         One of "train", "val", or "test".
        batch_size:    Samples per batch.
        num_workers:   Parallel data-loading workers.
        shuffle:       Shuffle the chosen split each epoch.
        seasons:       Seasons to include; None = all.
        num_channels:  Number of S2 bands (leading channels) to return.
        patch_size:    Random-crop spatial size; None = full 256×256.
        val_fraction:  Fraction of data reserved for validation.
        test_fraction: Fraction of data reserved for testing.
        seed:          Deterministic split seed.

    Returns:
        A PyTorch DataLoader whose batches are dicts::

            {
                "cloudy": (B, C, H, W) float32,
                "clean":  (B, C, H, W) float32,
                "sar":    (B, 2, H, W) float32,
            }
    """
    dataset = SEN12MSCRDataset(
        data_root,
        seasons=seasons,
        num_channels=num_channels,
        patch_size=patch_size,
        seed=seed,
    )

    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    if split == "train":
        subset_idx = indices[:n_train]
    elif split == "val":
        subset_idx = indices[n_train:n_train + n_val]
    elif split == "test":
        subset_idx = indices[n_train + n_val:]
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    subset = Subset(dataset, subset_idx.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# JAX bridge
# ---------------------------------------------------------------------------

def jax_iter(
    dataloader: DataLoader,
    n_devices: int,
    per_device_batch: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield batches formatted for JAX pmap.

    Converts PyTorch (B, C, H, W) float32 tensors to numpy arrays shaped
    (n_devices, per_device_batch, H, W, C) — channels-last, as JAX expects.

    Args:
        dataloader:        A DataLoader returned by :func:`get_dataloader`.
        n_devices:         Number of JAX/XLA devices (jax.local_device_count()).
        per_device_batch:  Samples per device per step.

    Yields:
        dict with keys "image" (cloudy) and "target" (clean), each shaped
        (n_devices, per_device_batch, H, W, C).
    """
    for batch in dataloader:
        cloudy = batch["cloudy"].numpy()  # (B, C, H, W)
        clean  = batch["clean"].numpy()

        # Channels-last: (B, C, H, W) → (B, H, W, C)
        cloudy = cloudy.transpose(0, 2, 3, 1)
        clean  = clean.transpose(0, 2, 3, 1)

        B = cloudy.shape[0]
        expected = n_devices * per_device_batch
        if B != expected:
            # Skip incomplete batches (drop_last=True should prevent this,
            # but guard against mismatched batch_size / device count).
            continue

        cloudy = cloudy.reshape(n_devices, per_device_batch, *cloudy.shape[1:])
        clean  = clean.reshape( n_devices, per_device_batch, *clean.shape[1:])

        yield {"image": cloudy, "target": clean}
