# CS159 Project: Experiments Overview

## Problem

Sentinel-2 imagery is frequently contaminated by clouds, blocking surface observations.
Existing cloud-removal models (e.g. EMRDM) restore plausible-looking images but do not
guarantee **spectral consistency** — the reconstructed reflectance values may not match
the physically correct spectral signatures of the underlying surface. This matters for
downstream tasks like NDVI computation, land-cover classification, and change detection,
which depend on accurate per-band reflectance.

The mirror-map approach addresses this by learning a bijection between the constrained
S2 reflectance space and an unconstrained mirror space. Any point in mirror space maps
back through `f_psi` to a spectrally valid S2 image, offloading the spectral constraint
from the generative model entirely.

---

## Shared Phase 1: Mirror Map Training

Both experiments use the same Phase 1 mirror map, trained on **real cloud-free S2 images**
from SEN12MS-CR (`ROIs1158_spring`).

- `g_phi` (ICNN gradient): maps cloud-free S2 reflectance → unconstrained mirror space
- `f_psi` (ResNet inverse map): maps mirror space → spectrally valid S2 reflectance

Training objective combines cycle consistency, a spectral constraint loss (SAM + per-band
moment matching), and an L1 regulariser encouraging `g_phi` to stay close to identity.

Best config from ablation: `spectral_sw0.1_mw1`
(sam_weight=0.1, moment_weight=1, dis_weight=0, style_weight=0)

| Metric | Value |
|--------|-------|
| Val loss | 0.0620 |
| Val MAE | 0.00898 |
| Val SAM | 0.0458 rad |
| Val PSNR | 36.77 dB |
| Val SSIM | 0.9648 |

Mirror-space properties (from `visualize_mirror_map.py` on cloud-free patches):
- `g_phi(x)` is systematically negative (mean ≈ −0.16, range ≈ [−0.49, 0.05])
- Round-trip MAE `f_psi(g_phi(x)) ↔ x`: 0.012 (close to val MAE, generalises well)
- Residuals are spatially smooth — no edge ringing or checkerboard artifacts
- Spectral profiles of reconstruction nearly overlap originals across all 13 bands

---

## Experiment 1: EMRDM (reflectance space) + Mirror Map as Post-Processing

### Pipeline

```
Cloudy S2 + S1  →  pretrained EMRDM  →  x_hat  (cloud-free, reflectance space)
                                              ↓
                                f_psi(g_phi(x_hat))  →  spectrally refined output
```

### Description

1. Use a pretrained EMRDM (trained on SEN12MS-CR in reflectance space) to generate
   cloud-free estimates `x_hat` from cloudy S2 + S1 SAR inputs.
2. Learn a mirror map on the **EMRDM-generated images** `x_hat`.
3. Apply the round-trip `f_psi(g_phi(x_hat))` to project EMRDM outputs onto the
   spectrally consistent manifold learned from those outputs.

### Key properties

- EMRDM is pretrained — no expensive training from scratch.
- Mirror map acts as post-hoc spectral correction, not as part of generation.
- **Risk**: the mirror map learns the manifold of EMRDM-generated images, not real
  cloud-free S2. If EMRDM outputs have spectral biases or artifacts, the mirror map
  encodes those rather than correcting them.

---

## Experiment 2: EMRDM Trained from Scratch in Mirror Space

### Pipeline

```
Cloud-free S2  →  g_phi  →  mirror-space targets  (training data for EMRDM)

At inference:
Cloudy S2 + S1  →  EMRDM (mirror space)  →  ỹ  →  f_psi  →  cloud-free S2
```

### Description

1. Apply pretrained `g_phi` to all cloud-free S2 training patches; store mirror-space
   TIFs (`prepare_mirror_dataset.py`).
2. Train EMRDM from scratch to predict mirror-space clean images, conditioned on
   [S1 SAR ∥ cloudy S2].
3. At inference: EMRDM samples a mirror-space estimate `ỹ`; `f_psi(ỹ)` projects it
   back to spectrally valid S2.

### Key properties

- Mirror map is learned on **real ground-truth cloud-free images** — correct manifold.
- Spectral constraints are enforced throughout generation, not just at the end.
- Requires training EMRDM from scratch (more compute than Exp 1).
- Diffusion mean = cloudy S2 in reflectance space (Option A); the model learns a
  residual that includes both the cloud-removal signal and the reflectance→mirror-space
  domain shift (~−0.16 mean offset).

### Implementation notes

- Mirror targets stored as `y × 10000` in TIF; `SEN12MSCRInterface` loads with `/10000`
  and `clip_target=False` (values are negative — clipping to [0,1] would zero them out).
- Config: `sentinel_mirror_train_scratch.yaml`

---

## Comparison

| | Baseline (EMRDM) | Exp 1 | Exp 2 |
|---|---|---|---|
| Mirror map trained on | — | EMRDM-generated images | Real cloud-free S2 |
| Spectral constraint | None | Post-hoc projection | During generation |
| EMRDM training | Pretrained | Pretrained | From scratch |
| Compute cost | Low | Low | High |
| Spectral manifold | None | Approximate (from generated data) | Correct (from real data) |

---

## Evaluation Metrics

Run all methods against held-out ground-truth cloud-free S2:

| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| SAM (rad) | ↓ | Per-pixel spectral fidelity — primary mirror-map claim |
| PSNR (dB) | ↑ | Pixel-level reconstruction accuracy |
| SSIM | ↑ | Structural similarity |
| MAE per band | ↓ | Band-specific spectral errors |
| NDVI error | ↓ | Vegetation index accuracy (physically interpretable) |

---

## Visualizations

**1. Side-by-side RGB panel**
Four columns per row: cloudy input | Exp 1 output | Exp 2 output | ground truth.
Select patches with thin cloud, thick cloud, and partial coverage.

**2. Spectral profiles**
Per-band mean reflectance for all four columns. If Exp 2's profile overlaps ground
truth more closely than Exp 1 or the baseline, mirror-space training is working.

**3. SAM heatmaps**
Per-pixel SAM as a false-color spatial map. Shows *where* each method fails spectrally
(cloud edges, thick cloud centers, shadows).

**4. Per-band error maps**
`(prediction − ground truth)` for representative bands B2, B4, B8, B11. Reveals
systematic spectral bias in specific wavelengths.

**5. Predicted vs. actual scatter plots**
One dot per pixel per band; well-calibrated model clusters around y=x diagonal.
Spectral bias appears as systematic offset.

---

## Expected Results

Exp 2 should win on SAM because the mirror map was learned on real data and spectral
constraints are enforced during generation. Exp 1 may show modest SAM improvement over
the baseline but is unlikely to match Exp 2, since the constraint manifold was learned
from imperfect generated data.

If Exp 1 and the baseline have similar SAM, that is a strong result supporting Exp 2:
it means post-hoc spectral correction alone is insufficient, and end-to-end mirror-space
training is necessary to achieve spectral consistency.

---

## Report Outline

1. **Introduction** — cloud contamination, spectral consistency, mirror-map motivation
2. **Related work** — EMRDM, NAMM, diffusion-based cloud removal
3. **Method** — Phase 1 mirror map (shared), Exp 1 pipeline, Exp 2 pipeline
4. **Experiments** — dataset, splits, implementation details, metrics
5. **Results** — quantitative table, RGB panels, spectral profiles, SAM heatmaps
6. **Analysis** — what the mirror map contributes, where each method fails
7. **Conclusion** — which approach works, compute trade-offs, future work (Phase 3)
