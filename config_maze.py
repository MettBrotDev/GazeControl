class Config:
    # 24x24 images: 6x6 blocks of 4x4 pixels
    IMG_SIZE = (60, 60)
    # Keep fovea/crop tiny for speed; single-scale is fine here
    FOVEA_OUTPUT_SIZE = (4, 4)      # encoder input size per scale
    FOVEA_CROP_SIZE = (4, 4)    # base crop; multi-scale uses 4,8, 16
    EPOCHS = 80

    # Training hyperparameters (small problem)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    BATCH_SIZE = 128

    # Model dims (lightweight)
    HIDDEN_SIZE = 512
    ENCODER_C1 = 12
    ENCODER_C2 = 16
    ENCODER_OUTPUT_SIZE = ENCODER_C2 * 4  # 2x2 pooled features
    POS_ENCODING_DIM = 64
    LSTM_LAYERS = 1

    # Multi-scale glimpses
    K_SCALES = 3                   # use 3 scales for maze
    FUSION_TO_DIM = 128
    FUSION_HIDDEN_MUL = 1.5

    # Decoder
    DECODER_LATENT_CH = 24
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Data: point at Maze dataset images
    MAZE_ROOT = "./Data/Maze15"
    LOCAL_DATA_DIR = [MAZE_ROOT]
    MNIST_DATA_DIR = "./Data/mnist"
    CIFAR100_DATA_DIR = "./Data/cifar100"

    # Rollout
    MAX_STEPS = 45
    MAX_MOVE = 0.1  # max gaze move per step (one tile)
    MIN_STEPS_BEFORE_STOP = 1
    USE_GAZE_BOUNDS = False  # allow exploration to image borders for maze start at top-left
    GAZE_BOUND_FRACTION = 0.05

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
    PRETRAIN_STEPS = 10000
    PRETRAIN_LR = 3e-3
    PRETRAIN_BATCH_SIZE = 128
    FREEZE_DECODER_EPOCHS = 1
    PRETRAINED_DECODER_PATH = "pretrained_components/pretrained_decoder_maze.pth"
    PRETRAIN_L1_WEIGHT = 1.0
    PRETRAIN_MSE_WEIGHT = 0.0
    PRETRAIN_USE_AMP = True
    PRETRAIN_PERC_WEIGHT = 0.002
    PRETRAIN_SSIM_WEIGHT = 0.0

    PRETRAINED_MODEL_PATH = ""  # not used
    # Warm-start for RL: set this to your recon-only checkpoint (e.g., gaze_control_model_local.pth)
    PRETRAINED_MODEL_PATH = "gaze_control_model_local.pth"

    # Use local Maze dir in training scripts
    DATA_SOURCE = "local"

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
    RL_STEP_PENALTY = 0.005 #maybe lower it to 0.005 for next training

    RL_ONLY_EPOCHS = 0

    # Initial gaze position control (normalized [0,1]): use top-left start with optional small jitter
    START_GAZE = (0.06, 0.06)    # set to None for random center start; (0,0) is exact corner
    START_JITTER = 0.02          # uniform jitter radius around START_GAZE; set 0.0 for fixed

    @classmethod
    def get_local_data_dirs(cls):
        if isinstance(cls.LOCAL_DATA_DIR, str):
            return [cls.LOCAL_DATA_DIR]
        return cls.LOCAL_DATA_DIR
