class Config:
    # Scale for ~300x300 Pathfinder images
    IMG_SIZE = (200, 200)               # downscale slightly from 300 
    FOVEA_OUTPUT_SIZE = (32, 32)        # higher encoder input res
    FOVEA_CROP_SIZE = (32, 32)          # base crop; multi-scale uses 32,64,128
    EPOCHS = 200

    # Training
    LEARNING_RATE = 5e-5        # Much slower LR to avoid collapse
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 0.5        # Tighter gradient clipping
    BATCH_SIZE = 32

    # Model dims
    HIDDEN_SIZE = 2048               # LSTM hidden size (was 3072 = 126M LSTM params!)
    ENCODER_C1 = 96                  # Wider encoder (was 64)
    ENCODER_C2 = 192                 # Wider encoder (was 128)
    ENCODER_OUTPUT_SIZE = ENCODER_C2 * 4  # due to AdaptivePool 2x2 = 768
    POS_ENCODING_DIM = 64
    LSTM_LAYERS = 2                  # 2 layers for better temporal modeling

    # Multi-scale glimpse
    K_SCALES = 3
    FUSION_TO_DIM = 1024             # Less compression: 3*768+64+2=2370 → 1024 (was 512)
    FUSION_HIDDEN_MUL = 2.0

    # Decoder
    DECODER_LATENT_CH = 192        # decoder capacity
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Data
    LOCAL_DATA_DIR = [
        "./Data/Pathfinder/curv_baseline/imgs",
        "./Data/Pathfinder/curv_contour_length_9/imgs",
        "./Data/Pathfinder/curv_contour_length_14/imgs",
    ]
    MNIST_DATA_DIR = "./Data/mnist"
    CIFAR100_DATA_DIR = "./Data/cifar100"

    # Rollout
    MAX_STEPS = 24
    MAX_MOVE = 0.1                 # smaller normalized moves at higher res
    USE_GAZE_BOUNDS = True
    GAZE_BOUND_FRACTION = 0.1

    # Reconstruction loss
    L1_WEIGHT = 1.0
    PERC_WEIGHT = 0.05             # MUCH stronger perceptual loss
    SSIM_WEIGHT = 2.0              # MUCH stronger structural loss
    
    # Foreground masking for L1 loss
    USE_FG_MASK = True             # Enable foreground-weighted L1 loss
    FG_THRESH = 0.15               # Higher threshold to focus on bright lines
    BG_WEIGHT = 0.01               # EXTREME: 100:1 ratio favoring foreground
    
    # Anti-collapse: variance penalty
    USE_VARIANCE_PENALTY = True    # Penalize uniform/low-variance outputs
    VARIANCE_PENALTY_WEIGHT = 0.5  # Strong penalty for collapse

    # Step loss
    USE_MASKED_STEP_LOSS = True
    STEP_LOSS_MIN = 0.05           # Increased from 0.02 - stronger early supervision
    STEP_LOSS_MAX = 0.5            # Increased from 0.35 - stronger late supervision
    FINAL_LOSS_MULT = 10.0         # Increased from 8.0 - emphasize final reconstruction
    STEP_MASK_SIGMA_SCALE = 0.35

    # Pretrained artifacts
    PRETRAIN_STEPS = 200000
    PRETRAIN_LR = 2e-3
    PRETRAIN_BATCH_SIZE = 16        # larger images -> smaller batch
    FREEZE_DECODER_EPOCHS = 0       # NEVER freeze - must adapt from start
    PRETRAINED_DECODER_PATH = "pretrained_components/pretrained_decoder_L.pth"
    PRETRAIN_L1_WEIGHT = 20.0
    PRETRAIN_MSE_WEIGHT = 0.0
    PRETRAIN_USE_AMP = True
    # Perceptual loss weight used only during decoder pretraining
    PRETRAIN_PERC_WEIGHT = 0.015
    # Structural term (1 - MS-SSIM) weight for decoder pretraining
    PRETRAIN_SSIM_WEIGHT = 6.0

    PRETRAINED_MODEL_PATH = "./PastRuns/precep_9M.pth"

    # Data source (use local Pathfinder dirs)
    DATA_SOURCE = "local"

    # RL
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

    # RL-first
    RL_ONLY_EPOCHS = 3

    @classmethod
    def get_local_data_dirs(cls):
        if isinstance(cls.LOCAL_DATA_DIR, str):
            return [cls.LOCAL_DATA_DIR]
        return cls.LOCAL_DATA_DIR
