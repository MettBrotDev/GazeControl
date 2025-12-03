class Config:
    # Snakes images are 128x128
    IMG_SIZE = (128, 128)
    # Slightly larger fovea for thin lines; multi-scale uses 16, 32, 64
    FOVEA_OUTPUT_SIZE = (16, 16)
    FOVEA_CROP_SIZE = (16, 16)
    EPOCHS = 200

    # Training hyperparameters (snakes, 128x128)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    BATCH_SIZE = 64

    # Model dims (tuned for 128×128 snakes)
    HIDDEN_SIZE = 768          # LSTM hidden/state size
    ENCODER_C1 = 32            # encoder channels stage 1
    ENCODER_C2 = 64            # encoder channels stage 2
    ENCODER_OUTPUT_SIZE = ENCODER_C2 * 4  # 2x2 pooled features
    POS_ENCODING_DIM = 64
    LSTM_LAYERS = 1

    # Multi-scale glimpses
    K_SCALES = 3                   # three scales work well at 128px
    FUSION_TO_DIM = 256
    FUSION_HIDDEN_MUL = 1.5

    # Decoder
    DECODER_LATENT_CH = 48
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Data roots
    MAZE_ROOT = "./Data/Maze10Random35"
    SNAKES_ROOT = "./Data/Snakes128"
    LOCAL_DATA_DIR = [SNAKES_ROOT]
    SNAKES_VAL_FRAC = 0.1  # train/val split for SnakesDataset
    MNIST_DATA_DIR = "./Data/mnist"
    CIFAR100_DATA_DIR = "./Data/cifar100"

    # Rollout
    MAX_STEPS = 45
    MAX_MOVE = 0.1  # max gaze move per step (one tile)
    MIN_STEPS_BEFORE_STOP = 15
    STOP_CONF_THRESH = 0.92  
    USE_GAZE_BOUNDS = True   # keep gaze safely within image for snakes
    GAZE_BOUND_FRACTION = 0.02

    # Reconstruction losses (L1-dominant). SSIM off for tiny images.
    L1_WEIGHT = 1.0
    PERC_WEIGHT = 0.0                
    SSIM_WEIGHT = 0.0
    GDL_WEIGHT = 0.0                # Useful for sharp edges in mazes

    # Step loss schedule and masking
    USE_MASKED_STEP_LOSS = True
    STEP_LOSS_MIN = 0.05
    STEP_LOSS_MAX = 0.5
    FINAL_LOSS_MULT = 8.0
    STEP_MASK_SIGMA_SCALE = 0.5
    # restrict final loss to regions the agent actually observed
    USE_FINAL_VISIBILITY_MASK = True
    # If not set, falls back to STEP_MASK_SIGMA_SCALE
    FINAL_MASK_SIGMA_SCALE = 0.5

    # Decoder Pretraining (if desired on Maze itself)
    PRETRAIN_STEPS = 15000
    PRETRAIN_LR = 3e-3
    PRETRAIN_BATCH_SIZE = 64
    FREEZE_DECODER_EPOCHS = 1
    PRETRAINED_DECODER_PATH = "pretrained_components/pretrained_decoder.pth"
    PRETRAIN_L1_WEIGHT = 1.0
    PRETRAIN_MSE_WEIGHT = 0.0
    PRETRAIN_USE_AMP = True
    PRETRAIN_PERC_WEIGHT = 0.01
    PRETRAIN_SSIM_WEIGHT = 0.0

    PRETRAINED_MODEL_PATH = ""  # not used
    # Warm-start for RL: set this to your recon-only checkpoint (e.g., gaze_control_model_local.pth)
    PRETRAINED_MODEL_PATH = "gaze_control_model_local.pth"

    # Select which dataset the main train.py uses: 'maze' or 'snakes'
    DATA_SOURCE = "snakes"

    # RL (defaults; not critical if you train supervised only)
    RL_GAMMA = 0.95
    RL_LAMBDA = 0.95
    RL_POLICY_LR = 3e-4
    RL_VALUE_COEF = 0.5
    RL_ENTROPY_COEF = 0.02
    RL_LOSS_WEIGHT = 1.0
    RL_CLIP_ACTION = True
    RL_INIT_STD = 0.2
    RL_REWARD_SCALE = 20.0
    RL_NORM_ADV = True
    RL_STOP_INIT_BIAS = -8.0      # stronger negative initialization for stop action
    # Small per-step time penalty to encourage stopping early when useful
    RL_STEP_PENALTY = 0.001 #maybe lower it to 0.005 for next training

    RL_ONLY_EPOCHS = 0

    # Initial gaze position control (used only if dataset doesn't provide one)
    START_GAZE = (0.5, 0.5)    # center spawn for snakes
    START_JITTER = 0.0         # fixed center

    @classmethod
    def get_local_data_dirs(cls):
        if isinstance(cls.LOCAL_DATA_DIR, str):
            return [cls.LOCAL_DATA_DIR]
        return cls.LOCAL_DATA_DIR