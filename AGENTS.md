# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repository is

A CS159 course project on **spectrally consistent cloud removal for Sentinel-2 imagery** using the SEN12MS-CR benchmark. Two distinct model families live in the tree — they are *not* combined into a single pipeline:

1. **NAMM mirror-map approach (Phase 1, primary, documented in README)** — `phase1_mirror_map/{icnn,inverse_map,losses,dataset,train_mirror_map}.py` and `phase1_mirror_map/train_mirror_map.slurm`. Trains a forward mirror map `g_phi` (ICNN gradient) and inverse map `f_psi` (ResNet) jointly so that a downstream EMRDM diffusion model can operate in an unconstrained mirror space while preserving spectral consistency on projection back. Phases 2 (EMRDM in mirror space) and 3 (inverse-map finetuning) are not yet present.
2. **PDE-constrained diffusion approach (exploratory, parked in `dump/`)** — `dump/pde_diffusion_model.py`, `dump/validate_sen12mscr.py`. A score-matching denoiser that adds Navier-Stokes and Maxwell PDE residual losses. Self-contained; does not import from the NAMM files. Treat it as a parallel experiment, not a stage of the NAMM pipeline.

The repo-root data downloader is `download_ROIs1158_spring.sh`. Other data utilities (`dump/dataLoader.py`, `dump/get_data.py`, `dump/dl_data.sh`, `dump/analysis.ipynb`) are not referenced by the documented Phase 1 pipeline and sit in `dump/` alongside the PDE track — keep them out of new Phase 1 code unless you have a reason to bring one back.

`phase2_emrdm/`, `phase3_finetune/`, `utils/`, and `tests/unit/{phase1,phase2,phase3}/` + `tests/integration/` exist as empty scaffolding (held open by `.gitkeep`) for the README's forthcoming phases. They contain no `__init__.py` — when code lands there, decide on package structure then.

## Environment

Reuses the EMRDM conda environment. From the README:

```bash
conda create --name emrdm python=3.10
conda activate emrdm
pip install torch==2.2.1 torchaudio==2.2.1 torchvision==0.17.1 numpy==1.26.4
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels
pip install pytorch-lightning==2.3.0
pip install wandb omegaconf rasterio tifffile scipy opencv-python lpips
```

There is no `requirements.txt`, no test suite, no linter config, and no `Makefile`. Verify changes by running training/validation directly.

## Data

`download_ROIs1158_spring.sh <dest>` fetches the ~70 GB spring ROI from `dataserv.ub.tum.de` over wget FTP (creds `m1554803:m1554803`, baked into the script), extracts each archive, and removes it. The archive list in that script is stale (`_s2_cloudfree.tar.gz` doesn't exist on the server — the real cloud-free archive is bare `_s2.tar.gz`); use `download_local_data.sh` for the working pattern. Phase 1 only needs the cloud-free `*_s2` tree (the TUM archives use the `_s2_cloudy` suffix for the cloudy variant; bare `_s2` is the cloud-free reference); `SEN12MSCRCloudFreeDataset` in `phase1_mirror_map/dataset.py` discovers patches by globbing `<data_root>/<roi>_s2/**/*.tif`. Each patch is 256×256 with 13 bands, normalised by dividing reflectance by 10000.

`dump/dl_data.sh` is an alternative interactive downloader for the full SEN12MS-CR / SEN12MS-CR-TS dataset.

`download_local_data.py` (run with `python download_local_data.py`) fetches a small **aligned-triplet** subset into `./data/` for local dev and tests. Streams all three Phase-1/2 archives — `ROIs1158_spring_s2.tar.gz` (cloud-free, anchor), `ROIs1158_spring_s1.tar.gz` (SAR), `ROIs1158_spring_s2_cloudy.tar.gz` (cloudy) — from `wget` FTP through Python's `tarfile` (streaming `r|gz` mode), extracting members one at a time so the ~13-34 GB tarballs never land on disk. The anchor pass takes `PER_SCENE` patches from each of `N_SCENES` distinct scenes of s2 (default 20 × 10 = 200, env-overridable: `PER_SCENE=10 N_SCENES=5 python download_local_data.py`), recording their `<scene>_p<N>` patch IDs; the s1 and s2_cloudy passes extract only patches whose ID is in that anchor set. A final reconciliation pass drops any patch missing from one of the three subsets. Final disk footprint is ~1.5 GB; bandwidth scales with `N_SCENES` (tar is sequential, so reaching N scenes streams through the leading N scenes of each archive). Streaming has no resume — a dropped connection restarts that archive. Writes a `data/.done` marker; the integration smoke test below is skipped when `data/` is absent. *(The older `download_local_data.sh` is left in the tree but superseded — use the `.py`.)*

## Common commands

**Phase 1 training (single GPU):**
```bash
cd phase1_mirror_map
python train_mirror_map.py \
    --data_root /scratch/$USER/cs159 \
    --output_dir /scratch/$USER/cs159/checkpoints/phase1 \
    --epochs 100 --batch_size 16 --fp16
```

**Resume:** add `--resume <path-to-ckpt>.pt`.

**SLURM:** `sbatch phase1_mirror_map/train_mirror_map.slurm` — edit `CODE_DIR`, `DATA_ROOT`, `OUTPUT_DIR`, partition, and email first. The script's `CODE_DIR` is an absolute deployment path (`…/code/phase1_mirror_map/`).

**Validation (PDE-diffusion track, parked):** `python dump/validate_sen12mscr.py …` (see argparse in the file). Expects directory layout `<root>/test/{s1,s2_cloudy,s2_cloudfree}/*.tif`.

**Phase 1 smoke test** (online wandb; downloads VGG16 weights once on first run, ~528 MB cached under `~/.cache/torch`):
```bash
python download_local_data.py      # one-time: fetch ~200 patches into data/
pytest tests/integration/test_train_mirror_map.py -v
# or, without pytest installed:
python tests/integration/test_train_mirror_map.py
```
Lives at `tests/integration/test_train_mirror_map.py`. Points `--data_root` at `<repo>/data/`, uses `--patch_size 64` to random-crop the real 256×256 patches (matching the official NAMM mirror-map resolution), runs 10 tiny epochs with `--wandb` enabled (run name `test_train_mirror_map` in the `cs159` project), and asserts `best.pt`, `last.pt`, `history.csv`, and the wandb summary contain all expected metric keys. It passes a large `--log_every` so metrics are logged once per epoch (one train + one val row per epoch in `history.csv`) rather than per step. It monkeypatches the dataset to record every patch file loaded and asserts all of `data/` is used (10 epochs amortise the `drop_last`-dropped patch). **Skips cleanly when `data/` is absent.** **Requires `wandb login` and network access** — fails informatively otherwise.

**Phase 1 unit tests** (CPU-only, no network):
```bash
pytest tests/unit/phase1 -v
```

**Wandb-enabled training** (production):
```bash
cd phase1_mirror_map
python train_mirror_map.py --data_root <DATA> --output_dir <OUT> \
    --wandb --wandb_project cs159 --wandb_run_name my-run
```

## Architecture notes worth knowing before editing

- **`ICNN` in `phase1_mirror_map/icnn.py` enforces convexity by clamping `Wz` weights non-negative with `F.relu(self.Wz.weight)` at every forward pass.** Replacing the leaky-ReLU activation with anything having a negative slope < 0 would break convexity — the constructor asserts `negative_slope >= 0`. The strong-convexity term `(alpha/2) ‖x‖²` is added inside `ICNN.potential`, not in the loss; the `strong_convexity` arg passed through `ICNNLayer` is currently unused (the constraint lives at the potential level).
- **`ICNNGradient.forward` wraps its body in `with torch.enable_grad():`** so callers may safely apply `torch.no_grad()` around the mirror map. `x.requires_grad_(True)` alone does **not** re-enable graph tracking inside an outer `no_grad` block — without `enable_grad`, `phi` has no `grad_fn` and `torch.autograd.grad` raises. Validation relies on this: it's not wrapped in `no_grad` itself (was removed when this issue surfaced), but `enable_grad` keeps the contract local to `ICNNGradient`.
- **`namm_loss` in `phase1_mirror_map/losses.py`** orchestrates three losses: L1 cycle (`x → y → x` *and* `y → x → y`), `ConstraintLoss` on noisy-mirror reconstructions (samples `sigma ~ U[0, max_sigma]` per-example), and an L1 regulariser on the implied ICNN gradient `(y - α x)/(1 - α)`. The constraint loss internally runs `x_hat` and `x_ref` through a frozen VGG16 with a learned 13→3 band projection initialised to pick S2 bands B4/B3/B2 as RGB.
- **`InverseMap` uses `GroupNorm(1, C)` = instance norm** and a final `ReLU` (reflectances are non-negative). The `residual=True` default adds the mirror-space input to the decoded output — relevant when interpreting reconstructions.
- **Two optimisers** (one each for `g_phi`, `f_psi`), each with EMA shadows (`ema_rate=0.999`). EMA tracking is per-parameter on `state_dict`; checkpoints persist the shadows.
- **Mixed precision via `torch.cuda.amp` (`GradScaler` + `autocast`).** This is the older `torch.cuda.amp` API, not `torch.amp` — keep it consistent if editing.

## Conventions observed in the code

- Sentinel-2 patches are `(13, 256, 256)` `float32` tensors in `[0, 1]` reflectance after the `/10000` clip.
- Checkpoints: `best.pt` (lowest val loss) + `last.pt` (overwritten every epoch). Both hold `g_phi`, `f_psi`, both optimisers, both EMAs, epoch, and `best_val_loss`. (Previous `final.pt` and `ckpt_ep{NNNN}.pt` are no longer written; `--save_every` was removed.)
- Per-step + per-epoch metrics land in `<output_dir>/history.csv` with a `phase` column ('train' or 'val'). Columns: `phase,epoch,step,elapsed_s,loss,l_cycle,l_constr,l_reg,mae,sam,psnr,ssim`. Train rows populate `step`; val rows leave it empty.
- Reconstruction-quality metrics (MAE, SAM in radians, PSNR in dB, SSIM) live in `losses.py` via `reconstruction_metrics(x_recon, x)`. `namm_loss` exposes `x_recon = f_psi(g_phi(x))` in its returned dict so the metrics avoid a second forward pass.
- Wandb is opt-in via `--wandb`. Default project is `cs159`. Metrics namespaces: `train/*`, `val/*` (loss + 4 recon metrics each), plus `train/lr`, `train/epoch`, `val/best_loss`.
- The NAMM files do not import from the PDE-diffusion files or vice versa — keep that separation when adding code.
