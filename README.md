# Spectrally Consistent Cloud Removal via Neural Approximate Mirror Maps

This repository integrates the [EDM](https://doi.org/10.48550/arXiv.2206.00364) diffusion model with [Neural Approximate Mirror Maps (NAMMs)](https://github.com/berthyf96/namm) to enforce spectral consistency constraints during the generative process. The approach is evaluated on the [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/) benchmark.

## Background

We investigate whether combining the NAMM and EDM frameworks improves cloud removal performance on the SEN12MS-CR dataset. Specifically, we ask: can a learned constrained diffusion model better preserve the spectral properties of multispectral satellite imagery compared to an unconstrained baseline?

This project addresses that question by:

1. Learning a mirror map `g_phi` that transforms cloud-free multispectral images into an unconstrained space where a diffusion model can operate freely.
2. Training a standard EDM (Karras et al. 2022) in that unconstrained mirror space, conditioned on SAR + cloudy Sentinel-2.
3. Projecting samples back to the constrained (spectrally consistent) space via the learned inverse map `f_psi`.

We define a differentiable constraint distance function as a combination of three complementary terms operating at different levels of spectral and spatial fidelity:

$$\ell_{\text{constr}}(\hat{\mathbf{x}}, \mathbf{x}) = \mathcal{L}_{\text{recon}}(\hat{\mathbf{x}}, \mathbf{x}) + \lambda_{\text{SAM}}\,\mathcal{L}_{\text{SAM}}(\hat{\mathbf{x}}, \mathbf{x}) + \lambda_{\text{mom}}\,\mathcal{L}_{\text{moments}}(\hat{\mathbf{x}}, \mathbf{x})$$

where $\hat{\mathbf{x}} = f_\psi(\mathbf{g}_\phi(\mathbf{x}) + \sigma\mathbf{z})$ is the image recovered by the inverse map from a noisy mirror-space point, and $\mathbf{x}$ is the cloud-free reference. The three terms enforce local spatial fidelity (`L_recon`, pixel-wise MSE), spectral shape consistency (`L_SAM`, spectral angle mapper loss), and global reflectance statistics (`L_moments`, per-band mean and variance), encouraging images that are physically consistent with Sentinel-2 observations. NAMM's construction provides a guarantee that final samples satisfy the constraint set.

## Repository structure

```
.
├── README.md
├── CLAUDE.md
├── data
|   ├── download_all_data.py          # Download all SEN12MSCR data
|   ├── download_ROIs1158_spring.sh   # Download script for the starter dataset
├── phase1_mirror_map/                # Phase 1: train NAMM mirror maps
│   ├── icnn.py                       # Input-convex NN: forward mirror map g_phi
│   ├── inverse_map.py                # ResNet decoder: inverse mirror map f_psi
│   ├── losses.py                     # l_constr and full NAMM training objective
│   ├── dataset.py                    # SEN12MS-CR cloud-free dataloader
│   ├── train_mirror_map.py           # Phase 1 training script
│   └── train_mirror_map.slurm        # SLURM job submission script
├── phase2_emrdm/                     # Phase 2: train EDM in mirror space
│   ├── train_mirror_diffusion.py     # Main Phase 2 training script (standard EDM, conditioned on S1S2)
│   ├── train_mirror_diffusion.sh     # SLURM job submission for Phase 2 training
│   ├── run_mirror_diffusion.py       # Inference: denoise in mirror space, invert via f_psi
│   ├── run_mirror_diffusion_ema.sh   # SLURM eval script (uses EMA checkpoint)
│   ├── eval_emrdm.py                 # Builds all_{train,val,test}_paths.pkl split files
│   ├── prepare_mirror_dataset.py     # Applies g_phi to write mirror-space TIFs (optional pre-compute)
│   ├── prepare_mirror_dataset.sh     # SLURM script for above
│   ├── visualize_predictions.py      # Side-by-side PNG: [Cloudy | Predicted | GT]
│   ├── check_mirror_range.py         # Scans mirror data range to calibrate sigma_data
│   ├── diagnose_dataset.py           # Verifies pkl triplet counts across seasons
│   ├── train_raw_diffusion.py        # Baseline EDM without mirror space (for comparison)
│   └── sgm/                          # EMRDM model library (upstream, minimally modified)
├── phase3_finetune/                  # Phase 3: finetune inverse map (forthcoming)
├── utils/                            # shared code (empty for now)
├── tests/
│   ├── unit/{phase1,phase2,phase3}/
│   └── integration/
└── dump/                             # parked/exploratory code outside the documented pipeline
```

Phase 2 (EDM training in mirror space) 

Phase 3 (inverse map finetuning) scripts are forthcoming.

## Setup

### 1. Clone dependencies

```bash
git clone https://github.com/Ly403/EMRDM.git
cd EMRDM
```

### 2. Create the environment

Follow the EMRDM setup instructions, which install PyTorch 2.2.1, flash-attn, natten, and pytorch-lightning. All Phase 1 dependencies are included in that environment.

**Quick install — conda (Linux + CUDA 12.1):**
```bash
conda env create -f env.yml
conda activate emrdm
# Final manual step — needed only for EMRDM Phase 2:
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
```

**Quick install — pip only (Phase 1 dev, macOS- or Linux-friendly):**
```bash
pip install '.[test]'
```
Skips `flash_attn` and `natten` (CUDA-only, Phase 2 only). Includes `pytest` for the smoke test.

**Manual install** (what `env.yml` encodes — useful for troubleshooting):
```bash
conda create --name emrdm python=3.10
conda activate emrdm
pip install torch==2.2.1 torchaudio==2.2.1 torchvision==0.17.1 numpy==1.26.4
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels
pip install pytorch-lightning==2.3.0
pip install wandb omegaconf rasterio tifffile scipy opencv-python lpips
```

### 3. Download the dataset

**Full dataset (cluster / training run, ~30 GB):**
```bash
cd data
chmod +x download_ROIs1158_spring.sh
./download_ROIs1158_spring.sh /scratch/$USER/cs159
```

This downloads ~30 GB (compressed) for the spring ROI: cloudy Sentinel-2, cloud-free Sentinel-2, and Sentinel-1 SAR. It extracts automatically and removes the archives to save space.

**Local subset (laptop dev / tests, ~1.5 GB):**
```bash
python data/download_local_data.py                        # 10 patches x 3 scenes
PER_SCENE=10 N_SCENES=5 python data/download_local_data.py # smaller / custom subset
```

Streams all three Phase-1/2 archives (cloud-free `_s2`, SAR `_s1`, cloudy `_s2_cloudy`) from `wget` FTP through Python's `tarfile`, extracting members one at a time so the multi-GB tarballs never land on disk. The anchor pass takes `PER_SCENE` patches from each of `N_SCENES` distinct scenes of the cloud-free archive (default 20 × 10 = 200), giving a scene-diverse subset; s1 and s2_cloudy are then filtered to those same patch IDs (`<scene>_p<N>`), and a reconciliation pass keeps only patches present in all three. Final disk footprint is ~1.5 GB (just the extracted patches). The integration smoke test under `tests/integration/` consumes the cloud-free subset and is skipped when it isn't present.

For the full 620 GB dataset, see the [SEN12MS-CR download page](https://patricktum.github.io/cloud_removal/sen12mscr/).

## Training pipeline

### Phase 1 — Train mirror maps

Trains the forward map `g_phi` (ICNN gradient) and inverse map `f_psi` (ResNet) jointly to minimise the NAMM objective.

**Interactive / single GPU:**
```bash
cd phase1_mirror_map
python train_mirror_map.py \
    --data_root /scratch/$USER/cs159 \
    --output_dir /scratch/$USER/cs159/checkpoints/phase1 \
    --epochs 100 \
    --batch_size 16 \
    --fp16
```

**SLURM cluster:**
```bash
# Edit the partition name and paths in the .slurm file first
sbatch phase1_mirror_map/stage2_coarse_sweep
```

**Resume from checkpoint:**
```bash
python train_mirror_map.py \
    --data_root /scratch/$USER/cs159 \
    --output_dir /scratch/$USER/cs159/checkpoints/phase1 \
    --resume /scratch/$USER/cs159/checkpoints/phase1/ckpt_ep0050.pt
```
We trained a sweep for 3 epochs, then we chose to train 3 configs for 100 epochs. 
- CONFIG1="spectral_sw0.1_mw1"
- CONFIG2="spectral_sw1_mw10"
- CONFIG3="spectral_sw10_mw10"

Key hyperparameters:

| Argument | Default | Description |
|---|---|---|
| `--max_sigma` | 0.1 | Max noise level for inverse map robustness |
| `--style_weight` | 100.0 | Weight for `L_style` in `l_constr` |
| `--cycle_weight` | 1.0 | Weight for cycle-consistency loss |
| `--constr_weight` | 1.0 | Weight for constraint distance loss |
| `--reg_weight` | 0.001 | ICNN sparsity regularisation weight |
| `--strong_convexity` | 0.3 | Strong-convexity coefficient alpha |

Checkpoints are saved to `--output_dir` every 10 epochs and whenever validation loss improves (`best.pt`).

### Phase 2 — Train EDM in mirror space

Trains EDM on mirror-space targets produced by `g_phi`. Uses the EDM entrypoint with a modified config that points to the mirror dataset and loads the Phase 1 checkpoint.

```bash
cd phase2_emrdm
python train_mirror_diffusion.py \
    --data_root /scratch/$USER/cs159 \
    --namm_ckpt /resnick/groups/perona/$USER/cs159/runs/stage3_top/spectral_sw0.1_mw1/best.pt \
    --output_dir /resnick/groups/perona/$USER/cs159/runs/phase2 \
```

Evaluate the EDM on the test dataset.

```bash
cd phase2_emrdm
python run_mirror_diffusion.py \
    --data_root  /resnick/groups/perona/$USER/cs159/data \
    --namm_ckpt  /resnick/groups/perona/$USER/cs159/runs/stage3_top/spectral_sw0.1_mw1/best.pt \
    --edm_ckpt   /resnick/groups/perona/$USER/cs159/runs/phase2/best.pt \
    --output_dir /resnick/groups/perona/$USER/cs159/output/mirror_diffusion \
    --split test --sampler heun --steps 40 --fp16
```

Visualizing the generated cloud-free samples from the test dataset.

```bash
cd phase2_emrdm
python phase2_emrdm/visualize_predictions.py \\
    --output_dir  /resnick/groups/perona/$USER/cs159/output/mirror_diffusion \\
    --data_root   /resnick/groups/perona/$USER/cs159/data \\
    --split       test \\
    --n_samples   12 \\
    --out_png     predictions_grid.png
```

### Phase 3 — Finetune inverse map (forthcoming)

Finetunes `f_psi` on actual EDM sampling errors to reduce distribution shift between the Gaussian noise used in Phase 1 training and the errors produced by the diffusion model.

Try training EMRDM in the mirror space.

### Training EDM model without NAMM

We train a EDM model without the NAMM mirror space constraints to compare how 
the two models perform.

## Evaluation

Evaluated on the SEN12MS-CR benchmark using:

- **MAE** &darr; (mean absolute error) — diff between predicted and ground truth
- **SAM** &darr; (spectral angle mapper) — spectral consistency
- **PSNR** &uarr; — pixel-wise reconstruction quality
- **SSIM** &uarr; (structural similarity index) — visual and spatial similarity

Results are directly comparable to published EMRDM and other SEN12MS-CR baselines.

## References

```
@inproceedings{feng2025neural,
  title={Neural Approximate Mirror Maps for Constrained Diffusion Models},
  author={Feng, Berthy T. and Baptista, Ricardo and Bouman, Katherine L.},
  booktitle={ICLR},
  year={2025}
}

@inproceedings{liu2025effective,
  title={Effective Cloud Removal for Remote Sensing Images by an Improved
         Mean-Reverting Denoising Model with Elucidated Design Space},
  author={Liu, Yi and Li, Wengen and Guan, Jihong and Zhou, Shuigeng and Zhang, Yichao},
  booktitle={CVPR},
  year={2025}
}

@article{yu2025satellitemaker,
  title={SatelliteMaker: A Diffusion-Based Framework for Terrain-Aware
         Remote Sensing Image Reconstruction},
  author={Yu, Zhenyu and Idris, Mohd Yamani Inda and Wang, Pei},
  journal={arXiv:2504.12112},
  year={2025}
}

@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  eprint={2206.00364},
  archivePrefix={arXiv},
}

@article{ebel2020multisensor,
  title={Multisensor Data Fusion for Cloud Removal in Global and
         All-Season Sentinel-2 Imagery},
  author={Ebel, Patrick and Meraner, Andrea and Schmitt, Michael
          and Zhu, Xiao Xiang},
  journal={IEEE TGRS},
  year={2020}
}
```