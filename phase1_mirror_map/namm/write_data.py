"""Save real data as .npy files. Besides GRMHD, all data is code-generated.
WARNING: This script will overwrite any previously written data.
"""
import argparse
import math
import os
import requests

import diffrax
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
from PIL import Image
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')  # use CPU-only
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import scipy.spatial.distance
from sklearn.gaussian_process.kernels import Matern

import pdes
from polish import simulation


parser = argparse.ArgumentParser()
parser.add_argument(
  '-d', '--dataset', type=str, default=None,
  choices=['riaf', 'burgers', 'kolmogorov', 'periodic', 'galaxies', 'sen12mscr'],)
parser.add_argument(
  '-f', '--filetype', type=str, default='tfrecord',
  choices=['tfrecord', 'npy'])
parser.add_argument(
  '-od', '--outdir', type=str, required=True, help='Base output directory.')
parser.add_argument(
  '-s', '--n_per_shard', type=int, default=100,
  help='Number of samples to save per .tfrecord or .npy file.')
parser.add_argument(
  '--n_train', type=int, default=10000, help='Total number of training samples.')
parser.add_argument(
  '--n_test', type=int, default=1000, help='Total number of test samples.')
parser.add_argument(
  '--n_val', type=int, default=1000, help='Total number of validation samples.')
parser.add_argument(
  '--seed', type=int, default=1, help='Random seed.')


_N_RIAF_SPIN_PARAMETERS = 101
_N_RIAF_INCLINATION_PARAMETERS = 90

# write_data.py — add this function alongside get_burgers_fns(), etc.

import glob
import rasterio  # or tifffile, or h5py depending on your file format

def get_sen12ms_cr_fns(data_root: str):
    """Load SEN12MS-CR patch triplets from disk.

    SEN12MS-CR directory layout (typical):
        <data_root>/
            ROIs<season>/
                S1/       # SAR, 2 bands, float32
                S2_cloudy/  # cloudy optical, 13 bands
                S2_cloudFree/  # target, 13 bands

    Each patch: 256 x 256 pixels.

    We pack each sample as a dict-style flat array with keys:
        s2_cloudy:    (256, 256, 13) float32  — input (cloudy optical)
        s2_cloudfree: (256, 256, 13) float32  — target (cloud-free)
        s1:           (256, 256, 2)  float32  — SAR (optional conditioning)

    For NAMM training, `image` = the cloud-free S2 patch (the constrained space).
    """
    # Gather all cloud-free patch paths
    cf_paths = sorted(glob.glob(
        os.path.join(data_root, '**', 'S2_cloudFree', '*.tif'), recursive=True))
    # Build parallel lists for cloudy and SAR
    def _cf_to_cloudy(p):
        return p.replace('S2_cloudFree', 'S2_cloudy')
    def _cf_to_s1(p):
        return p.replace('S2_cloudFree', 'S1').replace('S2_', 'S1_')

    def _read_patch(path, n_bands):
        with rasterio.open(path) as src:
            arr = src.read()          # (C, H, W)
        arr = arr.transpose(1, 2, 0) # (H, W, C)
        arr = arr[:256, :256, :n_bands].astype(np.float32)
        return arr

    def _normalize_s2(arr):
        """Clip to [0, 10000] reflectance range and scale to [0, 1]."""
        return np.clip(arr / 10000.0, 0.0, 1.0)

    def _normalize_s1(arr):
        """Log-scale SAR (dB), clip to [-25, 0] dB, scale to [0, 1]."""
        arr = np.clip(arr, 1e-6, None)
        arr_db = 10.0 * np.log10(arr)
        return np.clip((arr_db + 25.0) / 25.0, 0.0, 1.0)

    # Build a flat index once
    index = list(range(len(cf_paths)))

    def image_generating_fn(random_state):
        """random_state is a numpy RandomState; we draw the next index from it."""
        i = random_state.randint(0, len(cf_paths))
        cf_path = cf_paths[i]
        cloudy_path = _cf_to_cloudy(cf_path)

        cf = _normalize_s2(_read_patch(cf_path, 13))        # (256, 256, 13)
        cloudy = _normalize_s2(_read_patch(cloudy_path, 13))  # (256, 256, 13)

        # For NAMM: we train the mirror map on cloud-FREE images.
        # The cloudy image is the "corrupted" input used at inference.
        # Return cloud-free as the primary image; pack both for tfrecord.
        return {'cf': cf, 'cloudy': cloudy}

    def tf_example(sample):
        cf = sample['cf']       # (256, 256, 13)
        cloudy = sample['cloudy']
        h, w, c = cf.shape
        features = tf.train.Features(feature={
            'image':        tf.train.Feature(
                float_list=tf.train.FloatList(value=cf.reshape(-1).tolist())),
            'image_cloudy': tf.train.Feature(
                float_list=tf.train.FloatList(value=cloudy.reshape(-1).tolist())),
            'shape':        tf.train.Feature(
                int64_list=tf.train.Int64List(value=[h, w, c])),
        })
        return tf.train.Example(features=features)

    return image_generating_fn, tf_example


def get_riaf_fns():
  """Generate RIAF images of a black hole accretion disk.

  Each data sample has shape (h, w) = (100, 100).
  
  This is different from the proprietary GRMHD dataset used in the paper.
  Note that images are generated without a constant total brightness.
  The total brightness constraint is enforced during data preprocessing for
  training/eval.

  Images are pulled from http://vlbiimaging.csail.mit.edu/ by varying the
  spin and inclination parameters of the RIAF model of Sgr A*
  (Broderick et al., 2011). There are 101 possible spins and 90 possible
  inclination angles, amounting to 101 x 90 = 9090 total images.
  """
  def image_generating_fn(idxs_iterator):
    idx = next(idxs_iterator)
    s, i = np.unravel_index(
      idx, (_N_RIAF_SPIN_PARAMETERS, _N_RIAF_INCLINATION_PARAMETERS))
    url = f'http://vlbiimaging.csail.mit.edu/static/data/targetImgs/sgraBroderick/pmap_bs_{s:03d}_{i:03d}_2.png'
    data = requests.get(url).content
    # NOTE: The filepath might need to be changed to a valid directory.
    f = open('/tmp/img.png','wb')
    f.write(data)
    f.close()
    image = Image.open('/tmp/img.png')
    image = np.array(image)
    image = image / 255
    return image

  def tf_example(image):
    shape = image.shape
    features = tf.train.Features(
      feature={
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(image.reshape(-1)))),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1]]))
      }
    )
    example = tf.train.Example(features=features)
    return example

  return image_generating_fn, tf_example


def get_burgers_fns():
  """Generate displacements satisfying 1D Burgers' PDE for random initial conditions.
  
  Each data sample has shape (nx, nt).
  """
  mu = 1
  nu = 0.05

  x0 = 0
  x1 = 10
  nx = 64

  t0 = 0
  dt = 0.025
  inner_steps = 5
  nt = 64
  t1 = t0 + dt * inner_steps * (nt - 1)

  solver = pdes.CrankNicolson(rtol=1e-3, atol=1e-3)
  stepsize_controller = diffrax.ConstantStepSize()
  pde = pdes.BurgersEquation(
    mu, nu, x0, x1, nx, t0, t1, dt, nt, solver=solver, stepsize_controller=stepsize_controller)
  
  def _matern_cov(nx, cov_scale=1.):
    """Return the Matern covariance matrix for a square 2D grid."""
    kernel = Matern(length_scale=cov_scale)
    x, = np.meshgrid(np.linspace(-1, 1, nx))
    cov = kernel(X=np.expand_dims(x, axis=-1))
    return cov

  def _random_sample(mean, cov, random_state=None, **kwargs):
    if random_state is None:
      random_state = np.random.RandomState(np.random.randint(1000))
    w = random_state.multivariate_normal(mean, cov, **kwargs)
    return w

  def _normalize(data):
    """Normalize data to be in [0, 1]."""
    vmin = data.min()
    vmax = data.max()
    return (data - vmin) / (vmax - vmin)
  
  def _step(y0, t0):
    def _scan_fn(ycurr, tcurr):
      ynext, _, _, _, _ = solver.step(
        terms=diffrax.ODETerm(pde.dydt),
        t0=tcurr,
        t1=tcurr + dt,
        y0=ycurr,
        args=[],
        solver_state=None,
        made_jump=None)
      return ynext, ynext
    
    ts = jnp.linspace(t0, t0 + dt * inner_steps, inner_steps)
    y1, _ = jax.lax.scan(_scan_fn, y0, ts)
    return y1

  ts = jnp.linspace(t0, t0 + dt * inner_steps * nt, nt)
  @jax.jit
  def solve(y0, t0, nt):
    def _scan_fn(ycurr, tcurr):
      ynext = _step(ycurr, tcurr)
      return ynext, ynext
    _, y1 = jax.lax.scan(_scan_fn, y0, ts)
    return y1
  
  cov = _matern_cov(nx, cov_scale=0.2)

  def image_generating_fn(random_state):
    ic = _random_sample(np.ones(nx), cov, random_state)
    ic = _normalize(ic)
    image = solve(ic, t0, nt).T
    image = np.flipud(image)
    return image

  def tf_example(image):
    shape = image.shape
    features = tf.train.Features(
      feature={
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(image.reshape(-1)))),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1]]))
      }
    )
    example = tf.train.Example(features=features)
    return example
  
  return image_generating_fn, tf_example


def get_kolmogorov_fns():
  """Generate velocity fields for Kolmogorov flow with random initial conditions.
  
  Each data sample has shape (nt, ny, nx, 2).
  """
  size = 64
  dt = 0.01
  max_velocity = 3.
  reynolds = 1e3
  density = 1.
  inner_steps = 20
  snapshot_dt = inner_steps * dt
  t0 = 3
  n_snapshots = 8
  forcing = 'kolmogorov'

  grid = cfd.grids.Grid(
      shape=(size, size),
      domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )

  if forcing == 'kolmogorov':
    force = cfd.forcings.simple_turbulence_forcing(
      grid=grid,
      constant_magnitude=1.,
      constant_wavenumber=4.,
      linear_coefficient=-0.1,
      forcing_type='kolmogorov',
    )
  elif forcing == 'none':
    force = None
  else:
    raise ValueError('forcing not recognized')

  dt_stable = cfd.equations.stable_time_step(
    grid=grid,
    max_velocity=max_velocity,
    max_courant_number=0.5,
    viscosity=1 / reynolds
  )
  assert dt < dt_stable

  t_start = int(t0 / snapshot_dt) * snapshot_dt
  t_end = t_start + (snapshot_dt * n_snapshots)  # save n states from t_start
  outer_steps = int(np.ceil(t_end / snapshot_dt))

  t_start_idx = int(t_start / snapshot_dt)
  t_end_idx = outer_steps
  t_idxs = slice(t_start_idx, t_end_idx)
  n_states = t_end_idx - t_start_idx
  print(f'Saving {n_states} snapshots from time {t_start} to {t_end}')

  step_fn = cfd.funcutils.repeated(
    f=cfd.equations.semi_implicit_navier_stokes(
      grid=grid,
      forcing=force,
      dt=dt,
      density=density,
      viscosity=1 / reynolds,
    ),
    steps=inner_steps
  )
  rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps, start_with_input=True))

  def _sample_trajectory(rng):
    v0 = cfd.initial_conditions.filtered_velocity_field(
      rng,
      grid=grid,
      maximum_velocity=max_velocity,
      peak_wavenumber=4.
    )
    _, trajectory = rollout_fn(v0)
    return trajectory
  
  def _make_volume_from_trajectory(traj):
    u_traj = traj[0].array.data[t_idxs]  # (nt, size, size)
    v_traj = traj[1].array.data[t_idxs]
    return jnp.stack((u_traj, v_traj), axis=-1)

  def image_generating_fn(rng):
    volume = None
    while volume is None:
      trajectory = _sample_trajectory(rng)
      volume = np.ascontiguousarray(_make_volume_from_trajectory(trajectory))

      if np.isnan(volume).any() or np.isinf(volume).any() or np.isinf(-volume).any():
        continue

    return volume

  def tf_example(volume):
    shape = volume.shape
    features = tf.train.Features(
      feature={
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(volume.reshape(-1)))),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1], shape[2], shape[3]])),
        'reynolds': tf.train.Feature(float_list=tf.train.FloatList(value=[reynolds]))
      }
    )
    example = tf.train.Example(features=features)
    return example

  return image_generating_fn, tf_example


def get_periodic_fns():
  def _get_periodic_kernel(npix, sigma_f=0.7, sigma_l=0.7, period=(1, 1)):
    xx = np.linspace(0, 1, npix)
    yy = np.linspace(0, 1, npix)
    [X, Y] = np.meshgrid(xx, yy)
    points = np.zeros((npix * npix, 2))
    points[:, 0] = X.reshape(-1)
    points[:, 1] = Y.reshape(-1)
    
    points_i = points.reshape(npix * npix, 1, 2)
    points_j = points.reshape(npix * npix, 1, 2)
    displacements = points_i - np.transpose(points_j, (1, 0, 2))
    
    sin_arg1 = math.pi * np.abs(displacements[:, :, 0] / period[0])
    C1 = sigma_f**2 * np.exp(-2 * np.square(np.sin(sin_arg1)) / sigma_l**2)
    sin_arg2 = math.pi * np.abs(displacements[:, :, 1] / period[1])
    C2 = sigma_f**2 * np.exp(-2 * np.square(np.sin(sin_arg2)) / sigma_l**2)
    C = np.multiply(C1, C2)
    return C

  image_size = 64
  tile_size = 32
  n_periods = image_size // tile_size

  C = _get_periodic_kernel(tile_size)
  mu = np.ones(C.shape[0]) * 0.5

  def image_generating_fn(random_state):
    # TODO: It would be more efficient to draw all tile samples at once, i.e.,
    # with `random_state.multivariate_normal(mu, C, size=(n_per_shard,))`.
    tile = random_state.multivariate_normal(mu, C)
    tile = tile.reshape(tile_size, tile_size)
    tile = np.clip(tile, 0, 1)
    image = np.tile(tile, (n_periods, n_periods))
    image = np.piecewise(image, [image < 0.5, image >= 0.5], [0, 1])
    image = np.expand_dims(image, axis=-1)
    return image

  def tf_example(image):
    shape = image.shape
    features = tf.train.Features(
      feature={
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(image.reshape(-1)))),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1]])),
        'periods': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_periods]))
      }
    )
    example = tf.train.Example(features=features)
    return example

  return image_generating_fn, tf_example


def get_galaxies_fns():
  image_size = 128
  pixel_size = 0.25  # uas
  freq = 0.7
  background = True
  density = 1e6
  nsrc = 8
  snr_thresh = 15
  dist_thresh = 5
  SimObj = simulation.SimRadioGal(
    nx=image_size, ny=image_size, pixel_size=pixel_size, freqmin=freq, freqmax=freq, src_density_sqdeg=density)

  def simulate():
    if background:
      noise = np.random.normal(size=(image_size, image_size, 1)) * 8.5
      noise = gaussian_filter(noise, 4)
    min_flux = np.sqrt(np.var(noise) * 10**(snr_thresh / 10))
    # Note: x corresponds to vertical axis; y corresponds to horizontal axis.
    signal, peaks, coords = SimObj.sim_sky(nsrc=nsrc, distort_gal=False, min_flux=min_flux)
    # Zero-mean signal.
    peaks = peaks - np.mean(signal)
    signal = signal - np.mean(signal)

    # SNR of blobs.
    snr = 10 * np.log10(np.square(peaks) / np.var(noise))

    # Pairwise distances between blob centers.
    dists = scipy.spatial.distance.pdist(coords)

    return signal, noise, snr, dists

  def _normalize_data(data):
    data = data - data.min()
    data = data / data.max()
    return data

  def image_generating_fn(_):
    image = None
    while image is None:
      signal, noise, _, dists = simulate()

      if not np.all(dists > dist_thresh):
        continue

      image = signal + noise
      image = _normalize_data(image)

    return image

  def tf_example(image):
    shape = image.shape
    features = tf.train.Features(
      feature={
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(image.reshape(-1)))),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1]])),
        'nsrc': tf.train.Feature(int64_list=tf.train.Int64List(value=[nsrc]))
      }
    )
    example = tf.train.Example(features=features)
    return example
  
  return image_generating_fn, tf_example


if __name__ == '__main__':
  args = parser.parse_args()

  datadir = os.path.join(args.outdir, args.dataset)
  os.makedirs(datadir, exist_ok=True)

  if args.dataset == 'riaf':
    image_generating_fn, tf_example = get_riaf_fns()
    use_numpy_random_state = True
    if args.n_train + args.n_test + args.n_val != (_N_RIAF_SPIN_PARAMETERS * _N_RIAF_INCLINATION_PARAMETERS):
      raise ValueError('n_train + n_test + n_val must equal 101 x 90 = 9090.')
  elif args.dataset == 'burgers':
    image_generating_fn, tf_example = get_burgers_fns()
    use_numpy_random_state = True
  elif args.dataset == 'kolmogorov':
    image_generating_fn, tf_example = get_kolmogorov_fns()
    use_numpy_random_state = False
  elif args.dataset == 'periodic':
    image_generating_fn, tf_example = get_periodic_fns()
    use_numpy_random_state = True
  elif args.dataset == 'galaxies':
    image_generating_fn, tf_example = get_galaxies_fns()
    # TODO: Galaxies dataset actually doesn't use any random key method.
    # It would be nice to make the randomness reproducible.
    use_numpy_random_state = False
  elif args.dataset == "sen12mscr":
    image_generating_fn, tf_example = get_sen12ms_cr_fns(args.data_root)
    use_numpy_random_state = True
    # import glob, rasterio  # pip install rasterio
    
    # ---------- download a subset ----------
    # SEN12MS-CR is hosted at: https://mediatum.ub.tum.de/1554803
    # Each ROI folder contains s2_cloudy/ and s2_cloudFree/ subfolders with .tif files.
    # For a quick subset, manually download 2-3 ROI season folders, or use the
    # IEEE DataPort link. Place them under args.data_dir (e.g. data/sen12mscr_raw/).
    # Structure expected:
    #   data/sen12mscr_raw/
    #     ROIs1158_spring/
    #       s2_cloudFree/
    #         ROIs1158_spring_s2_cloudFree_p*.tif
    
    # BANDS = [1, 2, 3, 7]   # B2(blue), B3(green), B4(red), B8(NIR) — 0-indexed
    # PATCH_SIZE = 64         # NAMM examples all use 64x64
    
    # def load_sen12mscr_patches(root_dir, max_patches=None):
    #     files = sorted(glob.glob(f"{root_dir}/**/s2_cloudFree/*.tif", recursive=True))
    #     patches = []
    #     for f in files:
    #         with rasterio.open(f) as src:
    #             img = src.read(indexes=[b+1 for b in BANDS])  # rasterio is 1-indexed
    #             img = img.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)
    #             # Sentinel-2 L1C reflectance range is roughly 0-10000
    #             img = np.clip(img / 10000.0, 0, 1)
    #             # Tile into 64x64 patches
    #             H, W, C = img.shape
    #             for i in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
    #                 for j in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
    #                     patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
    #                     patches.append(patch)
    #         if max_patches and len(patches) >= max_patches:
    #             break
    #     return np.array(patches[:max_patches])  # (N, 64, 64, 4)
    
    # all_patches = load_sen12mscr_patches(
    #     args.data_dir, max_patches=args.n_train + args.n_test + args.n_val)
    # np.random.shuffle(all_patches)
    # splits = {
    #     'train': all_patches[:args.n_train],
    #     'test':  all_patches[args.n_train:args.n_train + args.n_test],
    #     'val':   all_patches[args.n_train + args.n_test:],
    # }
    
    # # ---------- write TFRecords (same schema as existing datasets) ----------
    # for split_name, data in splits.items():
    #     outpath = os.path.join(args.outdir, f'sen12mscr_{split_name}')
    #     os.makedirs(outpath, exist_ok=True)
    #     for shard_idx in range(0, len(data), args.n_per_shard):
    #         shard = data[shard_idx:shard_idx + args.n_per_shard]
    #         shard_file = os.path.join(outpath, f'{shard_idx:05d}.tfrecord')
    #         with tf.io.TFRecordWriter(shard_file) as writer:
    #             for img in shard:
    #                 # Match the schema used by FFHQ/CelebAHQ in score_flow/datasets.py:
    #                 # features: 'shape' (int64 x3) and 'data' (bytes)
    #                 img_uint8 = (img * 255).astype(np.uint8)
    #                 # store as (C, H, W) to match the existing decode logic
    #                 img_chw = img_uint8.transpose(2, 0, 1)
    #                 feature = {
    #                     'shape': tf.train.Feature(
    #                         int64_list=tf.train.Int64List(
    #                             value=[img_chw.shape[0], img_chw.shape[1], img_chw.shape[2]])),
    #                     'data': tf.train.Feature(
    #                         bytes_list=tf.train.BytesList(value=[img_chw.tobytes()])),
    #                 }
    #                 writer.write(
    #                     tf.train.Example(features=tf.train.Features(feature=feature))
    #                     .SerializeToString())
    #     print(f"Wrote {len(data)} patches for split '{split_name}' → {outpath}")

  n_per_shard = args.n_per_shard
  filetype = args.filetype

  random_state = np.random.RandomState(args.seed)
  rng = jax.random.PRNGKey(args.seed)
  if args.dataset == 'riaf':
    # RIAF dataset is generated with an iterator through possible image
    # parameters rather than a random state.
    random_state = iter(
      np.random.RandomState(args.seed).permutation(_N_RIAF_SPIN_PARAMETERS * _N_RIAF_INCLINATION_PARAMETERS))

  for split in ['train', 'test', 'val']:
    if split == 'train':
      n_samples = args.n_train
    elif split == 'test':
      n_samples = args.n_test
    elif split == 'val':
      n_samples = args.n_val

    n_in_remainder_shard = n_samples % n_per_shard
    n_shards = int(n_samples / n_per_shard) + (1 if n_in_remainder_shard != 0 else 0)

    for shard in range(n_shards):
      if shard == n_shards - 1 and n_in_remainder_shard != 0:
        n = n_in_remainder_shard
      else:
        n = n_per_shard
      if filetype == 'tfrecord':
        shard_path = os.path.join(
          datadir, f'{args.dataset}-{split}.tfrecord-{shard:05d}-of-{n_shards:05d}')
        with tf.io.TFRecordWriter(shard_path) as writer:
          for _ in tqdm(range(n), desc=f'Shard {shard + 1}/{n_shards}'):
            # TODO: For legacy reasons, some datasets are generated with a NumPy
            # random state rather than a JAX random key. It would be nice to use
            # the JAX method for all datasets.
            if use_numpy_random_state:
              image = image_generating_fn(random_state)
            else:
              rng, step_rng = jax.random.split(rng)
              image = image_generating_fn(step_rng)
            example = tf_example(image)
            writer.write(example.SerializeToString())
      elif filetype == 'npy':
        shard_path = os.path.join(datadir, f'set_{shard + 1:03d}.npy')
        shard_images = []
        for _ in tqdm(range(n), desc=f'{split} shard {shard + 1}/{n_shards}'):
          rng, step_rng = jax.random.split(rng)
          image = image_generating_fn(step_rng)
          shard_images.append(image)
        np.save(shard_path, np.array(shard_images))
        print(f'Saved {shard_path}')