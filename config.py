class Config:
    IMG_SIZE = (64, 64)                # Target image size for training images (smaller for CIFAR-100)
    FOVEA_OUTPUT_SIZE = (32, 32)       # Reduced for CIFAR scale
    FOVEA_CROP_SIZE = (48, 48)         # Reduced for CIFAR scale
    EPOCHS = 20                        # More epochs for CIFAR
    LEARNING_RATE = 1e-4               # Lower learning rate for spatial memory stability
    BATCH_SIZE = 1                     # No batching.
    HIDDEN_SIZE = 512                  # Reduced for spatial memory (state is simpler now)
    ENCODER_OUTPUT_SIZE = 128          # Reduced to match memory feature dim
    MEMORY_SIZE = (32, 32)             # Spatial memory grid size
    MEMORY_DIM = 3                     # Spatial memory channel count
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    # LOCAL_DATA_DIR can now be a string or list of strings
    LOCAL_DATA_DIR = ["./Data/Pathfinder/curv_baseline/imgs"
                      "./Data/Pathfinder/curv_contour_length_9/imgs",
                      "./Data/Pathfinder/curv_contour_length_14/imgs"]  # Can be ["path1", "path2", ...] or single string
    MNIST_DATA_DIR = "./Data/mnist"
    CIFAR100_DATA_DIR = "./Data/cifar100"   # Add CIFAR-100 data directory
    MAX_STEPS = 15                    # Reduced steps for CIFAR (smaller images)
    MAX_MOVE = 0.15                   # Slightly larger moves for smaller images
    EFFICIENCY_PENALTY = 0.005        # Reduced penalty
    GAMMA = 0.98
    DATA_SOURCE = "cifar100"          # "local", "mnist", or "cifar100"
    RECONSTRUCTION_WEIGHT = 3.0       # Reduced global weight (decoder is more direct now)
    FOVEAL_RECONSTRUCTION_WEIGHT = 50.0  # Reduced but still emphasized
    STEP_RECONSTRUCTION_DISCOUNT = 0.9   # Stronger discount for early good reconstructions
    VALUE_WEIGHT = 0.5               # Reduced since we're using simpler reward
    POLICY_WEIGHT = 1.0              # Keep policy weight balanced
    ENTROPY_WEIGHT = 0.1             # Increased entropy for spatial exploration
    REWARD_SCALE = 3.0               # Reduced for stability
    RL_LOSS_WEIGHT = 0.25             # Reduced to  focus on reconstruction first

    @classmethod
    def get_local_data_dirs(cls):
        """Returns LOCAL_DATA_DIR as a list, converting single string if needed."""
        if isinstance(cls.LOCAL_DATA_DIR, str):
            return [cls.LOCAL_DATA_DIR]
        return cls.LOCAL_DATA_DIR