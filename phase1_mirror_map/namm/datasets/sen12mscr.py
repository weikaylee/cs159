"""SEN12MS-CR dataset loader for NAMM training — pure numpy, no PyTorch.

Reads aligned triplets (cloudy S2, cloud-free S2, S1 SAR) directly from
.tif files using rasterio.  No PyTorch dependency, matching the official
NAMM environment (jax[cuda12] + tensorflow + flax etc.).

Value ranges assumed:
    S2 (cloudy and cloud-free): uint16, [0, 10000] reflectance ÷ 10000 → [0, 1]
    S1: float32, dB values in approximately [-25, 0] → linearly mapped to [0, 1]
        # TODO: verify S1 dB min/max on your specific dataset.

The public API matches what train_namm.py expects:
    loader = get_dataloader(data_root, split='train', batch_size=16, ...)
    for item in jax_iter(loader, n_devices, per_device_batch):
        item['image']   # cloudy,  (n_dev, per_dev_bs, H, W, C) float32
        item['target']  # clean,   (n_dev, per_dev_bs, H, W, C) float32
"""

import glob
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

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
# Triplet indexing (shared with verify_dataset.py)
# ---------------------------------------------------------------------------

def _read_tif(path: str) -> np.ndarray:
    """Return (C, H, W) float32 array from a GeoTIFF."""
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
    else:
        data = _tifffile.imread(path).astype(np.float32)
        if data.ndim == 2:
            return data[np.newaxis]
        if data.ndim == 3 and data.shape[2] <= data.shape[0]:
            return data.transpose(2, 0, 1)
        return data


def _index_dir(root: str, roi_num: str, season: str, modality: str) -> Dict:
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


def build_triplets(data_root: str, seasons: Optional[List[str]] = None) -> List[Dict]:
    """Scan data_root and return list of {clean, cloudy, sar} path dicts."""
    roi_re = re.compile(r'^ROIs(\d+)_(\w+)_s2$')
    triplets: List[Dict] = []

    try:
        all_dirs = os.listdir(data_root)
    except FileNotFoundError:
        return []

    for d in sorted(all_dirs):
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

        clean_idx  = _index_dir(clean_root,  roi_num, season, "s2")
        cloudy_idx = _index_dir(cloudy_root, roi_num, season, "s2_cloudy")
        sar_idx    = _index_dir(sar_root,    roi_num, season, "s1")

        for key in clean_idx:
            if key in cloudy_idx and key in sar_idx:
                triplets.append({
                    "clean":  clean_idx[key],
                    "cloudy": cloudy_idx[key],
                    "sar":    sar_idx[key],
                })

    return triplets


# ---------------------------------------------------------------------------
# Numpy DataLoader
# ---------------------------------------------------------------------------

class NumpyDataLoader:
    """Minimal numpy-based DataLoader — no PyTorch dependency.

    Yields dicts of numpy arrays:
        "cloudy": (B, C, H, W) float32
        "clean":  (B, C, H, W) float32
        "sar":    (2, H, W)    float32  (first sample only, for reference)
    """

    def __init__(
        self,
        triplets: List[Dict],
        batch_size: int = 16,
        num_channels: int = 13,
        patch_size: Optional[int] = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.triplets     = triplets
        self.batch_size   = batch_size
        self.num_channels = num_channels
        self.patch_size   = patch_size
        self.shuffle      = shuffle
        self._rng         = np.random.RandomState(seed)
        # expose .dataset attribute to match PyTorch DataLoader API
        self.dataset      = self

    def __len__(self) -> int:
        return len(self.triplets) // self.batch_size

    def _load_sample(self, triplet: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one triplet and return (cloudy, clean, sar) as (C, H, W) float32."""
        clean  = np.clip(_read_tif(triplet["clean"])  / S2_SCALE, 0.0, 1.0)
        cloudy = np.clip(_read_tif(triplet["cloudy"]) / S2_SCALE, 0.0, 1.0)
        sar    = np.clip(
            (_read_tif(triplet["sar"]) - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN),
            0.0, 1.0
        )

        if self.num_channels < clean.shape[0]:
            clean  = clean[:self.num_channels]
            cloudy = cloudy[:self.num_channels]

        if self.patch_size is not None:
            _, H, W = clean.shape
            top  = self._rng.randint(0, H - self.patch_size + 1)
            left = self._rng.randint(0, W - self.patch_size + 1)
            sl   = (slice(top, top + self.patch_size),
                    slice(left, left + self.patch_size))
            clean  = clean[:,  sl[0], sl[1]]
            cloudy = cloudy[:, sl[0], sl[1]]
            sar    = sar[:,    sl[0], sl[1]]

        return cloudy, clean, sar

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = np.arange(len(self.triplets))
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            cloudys, cleans = [], []
            for i in batch_idx:
                cloudy, clean, _ = self._load_sample(self.triplets[i])
                cloudys.append(cloudy)
                cleans.append(clean)

            yield {
                "cloudy": np.stack(cloudys),  # (B, C, H, W)
                "clean":  np.stack(cleans),
            }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 0,    # kept for API compatibility, unused
    shuffle: bool = True,
    seasons: Optional[List[str]] = None,
    num_channels: int = 13,
    patch_size: Optional[int] = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> NumpyDataLoader:
    """Return a NumpyDataLoader for the requested split.

    Args:
        data_root:     Path to the directory containing ROIs* folders.
        split:         One of "train", "val", "test".
        batch_size:    Samples per batch.
        num_workers:   Ignored (kept for backward compatibility).
        shuffle:       Shuffle each epoch.
        seasons:       Seasons to include; None = all.
        num_channels:  Number of S2 bands (leading channels).
        patch_size:    Random-crop size; None = full 256×256.
        val_fraction:  Fraction reserved for validation.
        test_fraction: Fraction reserved for testing.
        seed:          Deterministic split seed.
    """
    all_triplets = build_triplets(data_root, seasons=seasons)
    if not all_triplets:
        raise RuntimeError(
            f"No triplets found under {data_root!r}. "
            "Check the directory layout."
        )

    n = len(all_triplets)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_test  = max(1, int(n * test_fraction))
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    if split == "train":
        subset_idx = indices[:n_train]
    elif split == "val":
        subset_idx = indices[n_train:n_train + n_val]
    elif split == "test":
        subset_idx = indices[n_train + n_val:]
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    subset = [all_triplets[i] for i in subset_idx]
    return NumpyDataLoader(
        subset,
        batch_size=batch_size,
        num_channels=num_channels,
        patch_size=patch_size,
        shuffle=shuffle,
        seed=seed,
    )


def jax_iter(
    dataloader: NumpyDataLoader,
    n_devices: int,
    per_device_batch: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield batches shaped (n_devices, per_device_batch, H, W, C) for jax.pmap.

    Converts (B, C, H, W) → (n_devices, per_device_batch, H, W, C).
    """
    for batch in dataloader:
        cloudy = batch["cloudy"]  # (B, C, H, W)
        clean  = batch["clean"]

        B = cloudy.shape[0]
        if B != n_devices * per_device_batch:
            continue

        # Channels-last: (B, C, H, W) → (B, H, W, C)
        cloudy = cloudy.transpose(0, 2, 3, 1)
        clean  = clean.transpose(0, 2, 3, 1)

        cloudy = cloudy.reshape(n_devices, per_device_batch, *cloudy.shape[1:])
        clean  = clean.reshape( n_devices, per_device_batch, *clean.shape[1:])

        yield {"image": cloudy, "target": clean}
