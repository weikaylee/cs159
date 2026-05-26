"""
SEN12MS-CR dataset loader for Phase 1 mirror map training.

Phase 1 only requires cloud-free Sentinel-2 images (the constrained set).
SAR and cloudy images are not needed until Phase 2 (EMRDM training).

Expected directory layout after running the download script. The TUM
SEN12MS-CR archive `<roi>_s2.tar.gz` is the cloud-free reference (the
cloudy variant uses an explicit `_s2_cloudy` suffix), and unpacks to:

    <data_root>/
        ROIs1158_spring_s2/
            s2_<scene>/
                ROIs1158_spring_s2_<scene>_p<N>.tif

Each .tif is a 256x256 patch with 13 Sentinel-2 bands.

Note: when `roi=None`, the wildcard glob `*_s2` matches every cloud-free
ROI directory in `<data_root>` but would also pick up `_s2_cloudy/` if the
cloudy archive were extracted alongside. Pass an explicit `roi` (the
default `'ROIs1158_spring'` does this) to avoid the ambiguity.
"""

import os
import glob
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio


def _read_sentinel2_tif(path: str, normalize: bool = True) -> torch.Tensor:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    if normalize:
        img = img / 10000.0
        img = np.clip(img, 0.0, 1.0)
    return torch.from_numpy(img)


def _random_crop(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size is None or tensor.shape[-1] == patch_size:
        return tensor
    H, W = tensor.shape[1], tensor.shape[2]
    if H < patch_size or W < patch_size:
        raise ValueError(
            f'Patch size {patch_size} is larger than image size {(H, W)}'
        )
    i = random.randint(0, H - patch_size)
    j = random.randint(0, W - patch_size)
    return tensor[:, i:i + patch_size, j:j + patch_size]


class SEN12MSCRCloudFreeDataset(Dataset):
    """Cloud-free Sentinel-2 patches from SEN12MS-CR.

    Args:
        data_root:   Path to the dataset root (e.g. /scratch/oywang/cs159).
        split:       'train' or 'val'.
        val_fraction: Fraction of scenes held out for validation.
        seed:        Random seed for train/val split.
        normalize:   If True, normalise reflectance to [0, 1] by dividing
                     by 10000 (standard Sentinel-2 scale factor).
        patch_size:  Spatial size of output patches (None = use full 256x256).
        roi:         Specific ROI prefix to load, e.g. 'ROIs1158_spring'.
                     If None, loads all available ROIs.
    """

    S2_CLOUDFREE_GLOB = '*_s2'

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 val_fraction: float = 0.1,
                 seed: int = 42,
                 normalize: bool = True,
                 patch_size: Optional[int] = None,
                 roi: Optional[str] = 'ROIs1158_spring'):
        super().__init__()
        self.normalize = normalize
        self.patch_size = patch_size

        # Find all cloud-free .tif files
        if roi is not None:
            pattern = os.path.join(data_root, f'{roi}_s2',
                                   '**', '*.tif')
        else:
            pattern = os.path.join(data_root, self.S2_CLOUDFREE_GLOB,
                                   '**', '*.tif')

        all_files = sorted(glob.glob(pattern, recursive=True))
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No cloud-free .tif files found under {data_root}. "
                f"Check that the download completed and the directory "
                f"structure matches the expected layout."
            )

        # Deterministic train/val split by scene
        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_files) * val_fraction))
        if split == 'val':
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.files = [all_files[i] for i in selected]
        print(f"[SEN12MSCR] {split}: {len(self.files)} patches from {data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        with rasterio.open(path) as src:
            # (C, H, W) float32
            img = src.read().astype(np.float32)

        if self.normalize:
            img = img / 10000.0
            img = np.clip(img, 0.0, 1.0)

        tensor = torch.from_numpy(img)   # (13, H, W)

        if self.patch_size is not None and tensor.shape[-1] != self.patch_size:
            # Random crop
            H, W = tensor.shape[1], tensor.shape[2]
            i = random.randint(0, H - self.patch_size)
            j = random.randint(0, W - self.patch_size)
            tensor = tensor[:, i:i + self.patch_size, j:j + self.patch_size]

        return tensor


class SEN12MSCREMRDMOutputDataset(Dataset):
    """Paired ground truth and EMRDM-predicted Sentinel-2 patches.

    Args:
        data_root:   SEN12MS-CR root containing cloud-free `*_s2` folders.
        emrdm_root:  Root of EMRDM outputs, expected to contain
                     `predicted/*.tif` files with the same basename as the
                     cloud-free ground truth patches.
        split:       'train' or 'val'.
        val_fraction: Fraction of samples held out for validation.
        seed:        Random seed for train/val split.
        normalize:   If True, normalise reflectance to [0, 1].
        patch_size:  Random crop size.
        roi:         ROI prefix for ground truth files.
    """

    S2_CLOUDFREE_GLOB = '*_s2'

    def __init__(self,
                 data_root: str,
                 emrdm_root: str,
                 split: str = 'train',
                 val_fraction: float = 0.1,
                 seed: int = 42,
                 normalize: bool = True,
                 patch_size: Optional[int] = None,
                 roi: Optional[str] = 'ROIs1158_spring'):
        super().__init__()
        self.normalize = normalize
        self.patch_size = patch_size

        if roi is not None:
            gt_pattern = os.path.join(data_root, f'{roi}_s2', '**', '*.tif')
        else:
            gt_pattern = os.path.join(data_root, self.S2_CLOUDFREE_GLOB,
                                      '**', '*.tif')
        gt_files = sorted(glob.glob(gt_pattern, recursive=True))
        if len(gt_files) == 0:
            raise FileNotFoundError(
                f"No cloud-free .tif files found under {data_root}."
            )

        pred_files = sorted(glob.glob(os.path.join(emrdm_root, '**', '*.tif'),
                                      recursive=True))
        pred_files = [p for p in pred_files
                      if not p.endswith('_mu.tif')
                      and not p.endswith('_target.tif')]
        if len(pred_files) == 0:
            raise FileNotFoundError(
                f"No EMRDM predicted .tif files found under {emrdm_root}."
            )

        pred_map = {os.path.basename(p): p for p in pred_files}
        paired = []
        missing = []
        for gt in gt_files:
            name = os.path.basename(gt)
            if name in pred_map:
                paired.append((gt, pred_map[name]))
            else:
                missing.append(name)
        if len(paired) == 0:
            raise FileNotFoundError(
                f"No matching EMRDM outputs found for GT files under {data_root}."
            )
        if len(missing) > 0:
            print(f"[SEN12MSCREMRDM] Warning: {len(missing)} ground truth files "
                  f"did not match any EMRDM prediction and were skipped.")

        rng = random.Random(seed)
        indices = list(range(len(paired)))
        rng.shuffle(indices)
        n_val = max(1, int(len(paired) * val_fraction))
        if split == 'val':
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.pairs = [paired[i] for i in selected]
        print(f"[SEN12MSCREMRDM] {split}: {len(self.pairs)} paired patches "
              f"from {data_root} and {emrdm_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        gt_path, pred_path = self.pairs[idx]
        x_gt = _read_sentinel2_tif(gt_path, normalize=self.normalize)
        x_emrdm = _read_sentinel2_tif(pred_path, normalize=self.normalize)

        if self.patch_size is not None and x_gt.shape[-1] != self.patch_size:
            # Crop both inputs identically.
            H, W = x_gt.shape[1], x_gt.shape[2]
            i = random.randint(0, H - self.patch_size)
            j = random.randint(0, W - self.patch_size)
            x_gt = x_gt[:, i:i + self.patch_size, j:j + self.patch_size]
            x_emrdm = x_emrdm[:, i:i + self.patch_size, j:j + self.patch_size]

        return x_gt, x_emrdm


def build_dataloaders(data_root: str,
                      roi: str = 'ROIs1158_spring',
                      emrdm_root: Optional[str] = None,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      val_fraction: float = 0.1,
                      patch_size: Optional[int] = None,
                      seed: int = 42):
    """Build train and val DataLoaders for Phase 1, optionally with EMRDM outputs."""
    if emrdm_root is None:
        train_ds = SEN12MSCRCloudFreeDataset(
            data_root, split='train', val_fraction=val_fraction,
            seed=seed, patch_size=patch_size, roi=roi)
        val_ds = SEN12MSCRCloudFreeDataset(
            data_root, split='val', val_fraction=val_fraction,
            seed=seed, patch_size=patch_size, roi=roi)
    else:
        train_ds = SEN12MSCREMRDMOutputDataset(
            data_root, emrdm_root, split='train', val_fraction=val_fraction,
            seed=seed, patch_size=patch_size, roi=roi)
        val_ds = SEN12MSCREMRDMOutputDataset(
            data_root, emrdm_root, split='val', val_fraction=val_fraction,
            seed=seed, patch_size=patch_size, roi=roi)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
