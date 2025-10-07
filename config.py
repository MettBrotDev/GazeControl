class Config:
    IMG_SIZE = (32, 32)                # Target image size for training images
    FOVEA_OUTPUT_SIZE = (9, 9)         # All scales are resized to 9x9 for the encoder
    FOVEA_CROP_SIZE = (6, 6)           # Base crop size around gaze; multi-scale uses 6,12,24
    EPOCHS = 200

    # Training hyperparameters
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    BATCH_SIZE = 64                    # Use batching now
    HIDDEN_SIZE = 768                  # LSTM hidden size (matches autoencoder latent)
    ENCODER_OUTPUT_SIZE = 192          # Patch encoder output: 48*2*2 = 192 for 16x16 patch
    POS_ENCODING_DIM = 64              # Positional encoding size
    LSTM_LAYERS = 1                    # LSTM layers

    # Multi-scale glimpse settings
    K_SCALES = 3                       # number of scales: sizes base*(2**i)
    FUSION_TO_DIM = 256                # None to keep at k*E + pos + 2
    FUSION_HIDDEN_MUL = 2.0            # width multiplier for FusionMLP hidden layer

    # Decoder capacity and options
    DECODER_LATENT_CH = 96             
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    # LOCAL_DATA_DIR can be a string or list of strings
    LOCAL_DATA_DIR = [
        "./Data/Pathfinder/curv_baseline/imgs",
        "./Data/Pathfinder/curv_contour_length_9/imgs",
        "./Data/Pathfinder/curv_contour_length_14/imgs",
    ]
    MNIST_DATA_DIR = "./Data/mnist"
    CIFAR100_DATA_DIR = "./Data/cifar100"

    # Rollout (fixed 12 coverage positions)
    MAX_STEPS = 20
    MAX_MOVE = 0.2

    # Gaze movement bounds (keep gaze inside the central region to improve coverage)
    USE_GAZE_BOUNDS = True
    GAZE_BOUND_FRACTION = 0.1


    # Reconstruction loss weights (L1 dominant)
    MSE_WEIGHT = 0.1
    L1_WEIGHT = 1.0

    # Step loss schedule and masking
    USE_MASKED_STEP_LOSS = True        # Use a local mask around the current gaze for additional step losses
    STEP_LOSS_MIN = 0.02               # starting weight at step 1
    STEP_LOSS_MAX = 0.35                # ending weight at final step
    FINAL_LOSS_MULT = 8.0              # multiplier for final full-frame loss
    STEP_MASK_SIGMA_SCALE = 0.35       # scale the Gaussian sigma vs. patch size (smaller -> tighter)

    # Decoder Pretraining options (separate script uses these)
    PRETRAIN_STEPS = 40000          # total optimization steps
    PRETRAIN_LR = 3e-3             
    PRETRAIN_BATCH_SIZE = 32       
    FREEZE_DECODER_EPOCHS = 1      # freeze decoder for first epoch during main training
    PRETRAINED_DECODER_PATH = "pretrained_components/pretrained_decoder.pth"  # Path to pretrained decoder weights
    PRETRAIN_L1_WEIGHT = 1.0        # Try L1-only for extra sharpness
    PRETRAIN_MSE_WEIGHT = 0.0
    PRETRAIN_USE_AMP = True         # Mixed precision for faster/more stable pretraining
    # Perceptual loss weight used only during decoder pretraining
    PRETRAIN_PERC_WEIGHT = 0.005

    # Full-model checkpoint (e.g., a "random move" trained model)
    # If set to a valid path, training will load this entire state_dict first (strict=False),
    # then optionally override decoder with PRETRAINED_DECODER_PATH if requested.
    PRETRAINED_MODEL_PATH = "./PastRuns/rnd_10M.pth"      # e.g., "./PastRuns/WorkingIshModel0628.pth" or empty to disable

    # Data source
    DATA_SOURCE = "cifar100"          # "local", "mnist", or "cifar100"

    # RL settings (actor-critic on top of LSTM hidden state)
    RL_GAMMA = 0.95
    RL_LAMBDA = 0.95                 # for GAE advantage
    RL_POLICY_LR = 3e-4
    RL_VALUE_COEF = 0.5
    RL_ENTROPY_COEF = 0.02
    RL_LOSS_WEIGHT = 1.0             # scales the RL loss added to supervised loss
    RL_CLIP_ACTION = True            # clip delta to [-MAX_MOVE, MAX_MOVE] (useless in discrete case)
    RL_INIT_STD = 0.2                # initial std for Gaussian policy (in unscaled delta units)
    RL_REWARD_SCALE = 50.0          # multiply per-step rewards to strengthen signal
    RL_NORM_ADV = True               # normalize advantages to zero-mean, unit-std

    # Train RL first: freeze backbone
    RL_ONLY_EPOCHS = 3

    @classmethod
    def get_local_data_dirs(cls):
        """Returns LOCAL_DATA_DIR as a list, converting single string if needed."""
        if isinstance(cls.LOCAL_DATA_DIR, str):
            return [cls.LOCAL_DATA_DIR]
        return cls.LOCAL_DATA_DIR