"""Config for training NAMM on SEN12MS-CR cloud-removal with the sen_loss.

Usage (from the namm/ directory):

    python train_namm.py \
        --config configs/sen12mscr_config.py \
        --data_root /data/ \
        --workdir /path/to/checkpoints

Override any field on the command line with ml_collections syntax, e.g.:
    --config.training.batch_size=32
    --config.constraint.style_weight=200.0
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = config.data = ml_collections.ConfigDict()
    data.dataset       = "sen12mscr"
    # data_root is set at launch via --data_root; this is the default.
    data.data_root     = "/data/"
    # Spatial resolution fed to the model (random-cropped from 256×256).
    data.height        = 64
    data.width         = 64
    # Number of S2 bands to use.
    # 13 = all bands; 4 = B2/B3/B4/B8 subset.
    # TODO: set to 4 if you only want the RGB+NIR subset.
    data.num_channels  = 13
    # patch_size controls random crops inside the DataLoader.
    # Must equal data.height / data.width.
    data.patch_size    = 64
    data.num_workers   = 4
    data.random_flip   = True
    # Keep in [0, 1] — do not re-centre to [-1, 1].
    data.centered      = False
    data.uniform_dequantization = False
    # Unused by the sen12mscr path but required by other code that reads config.
    data.antialias             = True
    data.constant_flux         = False
    data.total_flux            = 120.0
    data.tfds_dir              = "./data"
    data.num_kolmogorov_states         = 8
    data.num_kolmogorov_states_per_row = 4
    data.kolmogorov_representation     = "image"

    # ------------------------------------------------------------------
    # Constraint  (sen_loss: reconstruction + distribution + style)
    # ------------------------------------------------------------------
    constraint = config.constraint = ml_collections.ConfigDict()
    constraint.type          = "sen12mscr"
    constraint.recon_weight  = 1.0
    constraint.dis_weight    = 1.0
    # Style weight is the dominant term; 100 matches the ConstraintLoss default.
    constraint.style_weight  = 100.0
    # Unused by sen12mscr but needed so other code doesn't KeyError.
    constraint.total_flux            = 120.0
    constraint.reynolds              = 1000.0
    constraint.inner_steps           = 20
    constraint.kolmogorov_dt         = 0.01
    constraint.kolmogorov_forcing    = True
    constraint.kolmogorov_t0         = 3
    constraint.burgers_t0            = 0
    constraint.burgers_dt            = 0.025
    constraint.burgers_inner_steps   = 5
    constraint.counting_cnn_ckpt_path = "./checkpoints/counting_cnn"

    # ------------------------------------------------------------------
    # Model  (ResNet fwd + ResNet bwd; ICNN not needed for paired setup)
    # ------------------------------------------------------------------
    model = config.model = ml_collections.ConfigDict()
    model.fwd_network        = "resnet"
    model.bwd_network        = "resnet"
    model.fwd_n_filters      = 64
    model.bwd_n_filters      = 64
    model.n_res_blocks       = 6
    model.dropout_rate       = 0.5
    model.n_downsample_layers = 2
    model.upsample_mode      = "deconv"
    model.fwd_residual       = True
    model.bwd_residual       = False
    model.fwd_activation     = "none"
    # relu keeps reconstructions non-negative (reflectance ≥ 0).
    model.bwd_activation     = "relu"
    model.fwd_strong_convexity = 0.9
    model.bwd_strong_convexity = 0.1
    model.fwd_icnn_n_filters = 32
    model.bwd_icnn_n_filters = 64
    model.fwd_icnn_n_layers  = 3
    model.bwd_icnn_n_layers  = 5
    model.fwd_icnn_kernel_size = 3
    model.bwd_icnn_kernel_size = 3
    model.ema_rate           = 0.999

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    optim = config.optim = ml_collections.ConfigDict()
    optim.learning_rate          = 2e-4
    optim.adam_beta1             = 0.5
    optim.grad_clip              = -1.0   # negative = no clipping
    optim.zero_nans              = False
    optim.cycle_weight           = 1.0
    optim.constraint_weight      = 0.1
    optim.regularization_weight  = 1e-3
    # fwdid regularisation: penalises ‖g_phi(x) - x‖ to keep the map near-identity.
    optim.regularization         = "fwdid"
    optim.max_sigma              = 0.1
    optim.fixed_sigma            = False
    optim.mdm_finetune           = False
    optim.dsm_weight             = 1e-3
    optim.divergence_weight      = 0.1

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    training = config.training = ml_collections.ConfigDict()
    training.batch_size           = 16
    training.n_epochs             = 100
    training.log_freq             = 50    # steps between log lines
    training.snapshot_epoch_freq  = 5     # save plot every N epochs
    training.ckpt_epoch_freq      = 10    # save checkpoint every N epochs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    config.eval = evaluation = ml_collections.ConfigDict()
    evaluation.batch_size = 16

    config.seed = 42

    return config
