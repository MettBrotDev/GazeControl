class Config:
    # Scale for ~300x300 Pathfinder images
    IMG_SIZE = (300, 300)               # downscale slightly from 300 for divisible sizes
    FOVEA_OUTPUT_SIZE = (24, 24)        # higher encoder input res
    FOVEA_CROP_SIZE = (24, 24)          # base crop; multi-scale uses 24,48,96
    EPOCHS = 200

    # Training
    LEARNING_RATE = 3e-4        # LOWER THE LR BEFORE CONTINUING A RL TRAINING RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    BATCH_SIZE = 32

    # Model dims
    HIDDEN_SIZE = 2048               # reasonable state for larger images
    ENCODER_C1 = 64                  # Wider encoder
    ENCODER_C2 = 128
    ENCODER_OUTPUT_SIZE = ENCODER_C2 * 4  # due to AdaptivePool 2x2
    POS_ENCODING_DIM = 64
    LSTM_LAYERS = 2

    # Multi-scale glimpse
    K_SCALES = 3
    FUSION_TO_DIM = 512          # fusion dimension
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
    MSE_WEIGHT = 0.1
    L1_WEIGHT = 1.0
    PERC_WEIGHT = 0.005

    # Step loss
    USE_MASKED_STEP_LOSS = True
    STEP_LOSS_MIN = 0.02
    STEP_LOSS_MAX = 0.35
    FINAL_LOSS_MULT = 8.0
    STEP_MASK_SIGMA_SCALE = 0.35

    # Pretrained artifacts
    PRETRAIN_STEPS = 40000
    PRETRAIN_LR = 2e-3
    PRETRAIN_BATCH_SIZE = 16        # larger images -> smaller batch
    FREEZE_DECODER_EPOCHS = 1
    PRETRAINED_DECODER_PATH = "pretrained_components/pretrained_decoder_L.pth"
    PRETRAIN_L1_WEIGHT = 1.0
    PRETRAIN_MSE_WEIGHT = 0.0
    PRETRAIN_USE_AMP = True

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
