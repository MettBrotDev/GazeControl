class Config:
    IMG_SIZE = (64, 64)               # Target image size for training images.
    FOVEA_OUTPUT_SIZE = (16, 16)
    FOVEA_CROP_SIZE = (32, 32)
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1                    # No batching.
    HIDDEN_SIZE = 256        #512
    ENCODER_OUTPUT_SIZE = 64
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    LOCAL_DATA_DIR = "./Data/labeled/test"  # Local dataset root
    MNIST_DATA_DIR = "./Data/mnist"
    MAX_STEPS = 20                   # Number of gaze movements per image (episode)
    MAX_MOVE = 0.1            # Limit movement to 20% of image dimensions
    EFFICIENCY_PENALTY = 0.02   # small negative reward per gaze step
    GAMMA = 0.98
    DATA_SOURCE = "mnist"             # "local" or "mnist"
    RECONSTRUCTION_WEIGHT = 10.0      # scale of reconstruction loss
    STEP_RECONSTRUCTION_DISCOUNT = 0.95  # discount factor for reconstruction loss per step
    VALUE_WEIGHT = 2.0           # scale of value loss
    POLICY_WEIGHT = 1.0            # scale of policy loss
    ENTROPY_WEIGHT = 0.01          # scale of entropy loss

    # ----- Warmup parameters -----
    PRETRAIN_SAMPLE_SIZE = 5000  # Number of samples for warmup
