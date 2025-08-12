import os
# silence TF/XLA plugin re‐registration messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # suppress TF logs
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import datetime
import numpy as np
from foveal_blur import foveal_blur
import torch.distributions as D
import glob
from PIL import Image

from models import GazeControlModel
from config import Config          

def clear_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def train():
    # timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"gaze_control_{Config.DATA_SOURCE}_{timestamp}"
    run_dir = os.path.join("runs", log_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)  # Initialize tensorboard writer
    
    # Track images seen and checkpoint frequency
    imgs_seen = 0
    next_save_at = 5000

    # Setup transforms
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
    ])
    transform_mnist = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1))
    ])

    # choose dataset
    if Config.DATA_SOURCE == "mnist":
        train_dataset = datasets.MNIST(root=Config.MNIST_DATA_DIR,
                                       train=True,
                                       transform=transform_mnist,
                                       download=True)
    elif Config.DATA_SOURCE == "cifar100":
        train_dataset = datasets.CIFAR100(root=Config.CIFAR100_DATA_DIR,
                                         train=True,
                                         transform=transform_img,
                                         download=True)
    else:
        # Handle multiple local data directories
        local_dirs = Config.get_local_data_dirs()
        local_datasets = []
        for data_dir in local_dirs:
            if os.path.exists(data_dir):
                dataset = SimpleImageDataset(data_dir, transform=transform_img)
                local_datasets.append(dataset)
                print(f"Added dataset from {data_dir} with {len(dataset)} images")
            else:
                print(f"Warning: Directory {data_dir} does not exist, skipping...")
        
        if not local_datasets:
            raise ValueError("No valid local data directories found!")
        
        # Combine all local datasets
        train_dataset = ConcatDataset(local_datasets) if len(local_datasets) > 1 else local_datasets[0]
        print(f"Total combined dataset size: {len(train_dataset)} images")

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)

    # Initialize model, loss and optimizer
    # Current Task: Full reconstruction with spatial memory.
    model = GazeControlModel(encoder_output_size=Config.ENCODER_OUTPUT_SIZE,
                             state_size=Config.HIDDEN_SIZE,
                             img_size=Config.IMG_SIZE,
                             fovea_size=Config.FOVEA_OUTPUT_SIZE,
                             memory_size=Config.MEMORY_SIZE,
                             memory_dim=getattr(Config, 'MEMORY_DIM', 4)).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    rec_loss_fn = torch.nn.MSELoss()  # Reconstruction loss

    model.train()
    for epoch in range(Config.EPOCHS):
        for batch_idx, (images, _) in enumerate(train_loader):
            total_rl_loss = torch.tensor(0.0, device=Config.DEVICE, requires_grad=True)
            total_rec_loss = torch.tensor(0.0, device=Config.DEVICE, requires_grad=True)    
            image = images.to(Config.DEVICE)
            # Initialize gaze randomly between 0.25 and 0.75
            gaze = torch.rand(1, 2, device=Config.DEVICE) * 0.5 + 0.25
            img_np = (image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            cx = int(gaze[0,0].item() * Config.IMG_SIZE[0])
            cy = int(gaze[0,1].item() * Config.IMG_SIZE[1])
            blurred = foveal_blur(img_np, (cx, cy),
                                  output_size=Config.FOVEA_OUTPUT_SIZE,
                                  crop_size=Config.FOVEA_CROP_SIZE)
            foveated_image = torch.from_numpy(blurred.astype(np.float32)/255.0)\
                                .permute(2,0,1).unsqueeze(0).to(Config.DEVICE)

            # start tracking gaze history to penalize repeats
            gaze_history = [gaze.squeeze(0)]

            # Initialize spatial memory
            memory = model.init_memory(1, Config.DEVICE)

            reconstruction = None

            # Perform a fixed number of gazemovements 
            for step in range(Config.MAX_STEPS):
                if step == 0:
                    # No previous rec_error and initialize next_value for TD target
                    prev_rec_error = torch.tensor(0.0, device=Config.DEVICE)
                # rl step (now also returns policy heads so we don't need another forward)
                reconstruction, (gaze_delta, stop_action), policy_bundle, memory = \
                    model.sample_action(foveated_image, memory, gaze)

                # Unpack policy heads
                state_value, gaze_mean, gaze_std, stop_logit, explore_mask = policy_bundle

                # compute new reconstruction error
                rec_error = rec_loss_fn(reconstruction, image) * Config.RECONSTRUCTION_WEIGHT



                # Regional reconstruction loss - focus on the foveal region with blur-aware weighting
                # We create a masked reconstruction loss that matches the blur pattern
                # This ensures that the region we look at gets through the network instead of getting lost (Weird way of putting that but makes sense in my mind)
                
                foveal_mask = torch.zeros_like(image)
                foveal_half_size = Config.FOVEA_CROP_SIZE[0] // 2
                
                # Calculate the bounds of the foveal region
                cx = int(gaze[0,0].item() * Config.IMG_SIZE[0])
                cy = int(gaze[0,1].item() * Config.IMG_SIZE[1])
                
                # Ensure bounds are within image
                x_start = max(0, cx - foveal_half_size)
                x_end = min(Config.IMG_SIZE[0], cx + foveal_half_size)
                y_start = max(0, cy - foveal_half_size)
                y_end = min(Config.IMG_SIZE[1], cy + foveal_half_size)
                
                # Create distance-based weighting that matches the blur pattern
                if x_end > x_start and y_end > y_start:
                    # Create coordinate grids for the foveal region
                    y_coords = torch.arange(y_start, y_end, device=Config.DEVICE).float()
                    x_coords = torch.arange(x_start, x_end, device=Config.DEVICE).float()
                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    
                    # Calculate distance from foveal center
                    distances = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                    max_distance = foveal_half_size
                    norm_distances = distances / max_distance
                    
                    # Create weight map that matches foveal_blur: high weight in center, low at edges
                    # This mimics the blur pattern from foveal_blur.py
                    fovea_radius_norm = 0.2  # Same as in foveal_blur.py
                    blur_strength = (norm_distances - fovea_radius_norm) / (1.0 - fovea_radius_norm)
                    blur_strength = torch.clamp(blur_strength, 0, 1)
                    
                    # Invert blur strength to get reconstruction weight (high where blur is low)
                    reconstruction_weight = 1.0 - blur_strength
                    
                    # Apply the weight to the mask
                    foveal_mask[:, :, y_start:y_end, x_start:x_end] = reconstruction_weight.unsqueeze(0).unsqueeze(0)
                
                # Compute weighted reconstruction loss for the foveal region
                weighted_reconstruction_diff = (reconstruction - image) * foveal_mask
                foveal_rec_error = torch.mean(weighted_reconstruction_diff ** 2)
                foveal_rec_loss = foveal_rec_error * Config.FOVEAL_RECONSTRUCTION_WEIGHT
                
                rec_loss = rec_error.clone() + foveal_rec_loss

                # penalize if new gaze is too close to any previous gaze
                dist_penalty = torch.tensor(0.0, device=Config.DEVICE)
                for prev_gaze in gaze_history:
                    dist = torch.norm(gaze - prev_gaze, dim=-1)
                    # penalize based on how small the min distance is
                    if dist.item() < 0.05:  # threshold for minimum distance
                        penalty = (0.05 - dist.item()) * 2.0  # scale penalty
                        dist_penalty += penalty

                # reward = how much error dropped minus cost of an extra glimpse
                # !!!!!!! WATCH OUT THIS IS ONLY FOR THIS TASK, if the Reward is caluclated differently its not useful anymore
                improvement = (prev_rec_error - rec_error)
                if step == 0:
                    # First step: don't scale improvement (it's artificially high)
                    reward = improvement.detach() - Config.EFFICIENCY_PENALTY - dist_penalty
                else:
                    # Subsequent steps: scale improvement and include distance penalty
                    reward = improvement.detach() * Config.REWARD_SCALE - Config.EFFICIENCY_PENALTY - dist_penalty

                prev_rec_error = rec_error.detach()

                # For the spatial memory model, we use the state value from sample_action
                td_target = reward  # Simple immediate reward for now

                # Value prediction is already computed in sample_action call above
                value_pred = state_value

                # 1) Critic loss
                value_loss = nn.functional.mse_loss(value_pred, td_target.detach())
                # 2) Actor (policy) loss - use heads from sample_action (single forward)
                gaze_dist = D.Normal(gaze_mean, gaze_std)
                stop_prob = torch.sigmoid(stop_logit)
                stop_dist = D.Bernoulli(stop_prob)

                logp_gaze = gaze_dist.log_prob(gaze_delta).sum(-1)      # sum dims dx,dy
                logp_stop = stop_dist.log_prob(stop_action)
                advantage = (td_target - value_pred).detach()
                # Zero-out gradients when exploring randomly
                explore_mask_flat = explore_mask.squeeze(1)
                policy_term = (1.0 - explore_mask_flat) * (logp_gaze + logp_stop)
                policy_loss = -policy_term * advantage

                # Entropy bonus for exploration
                ent_gaze = gaze_dist.entropy().sum(-1)  # sum dims dx,dy
                ent_stop = stop_dist.entropy()
                entropy = (ent_gaze + ent_stop).mean()

                # 3) Combined loss
                rl_loss = Config.VALUE_WEIGHT * value_loss \
                        + Config.POLICY_WEIGHT * policy_loss.mean() \
                        - Config.ENTROPY_WEIGHT * entropy               


                total_rl_loss = total_rl_loss + rl_loss
                total_rec_loss = total_rec_loss + rec_loss * np.power(Config.STEP_RECONSTRUCTION_DISCOUNT, (step + 1) / Config.MAX_STEPS)

                # Memory is already updated in the model.sample_action call
                # clamp and apply gaze_delta
                max_m = torch.tensor([Config.MAX_MOVE,
                                      Config.MAX_MOVE],
                                     device=Config.DEVICE, dtype=gaze_delta.dtype)
                gaze_delta = gaze_delta * max_m
                # Ensure gaze_delta is within bounds
                gaze = (gaze + gaze_delta).detach()
                gaze = torch.clamp(gaze, 0, 1)
                gaze_history.append(gaze.squeeze(0))
                # Update foveated_image based on new gaze
                img_np = (image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                cx = int(gaze[0,0].item() * Config.IMG_SIZE[0])
                cy = int(gaze[0,1].item() * Config.IMG_SIZE[1])
                blurred = foveal_blur(img_np, (cx, cy),
                                      output_size=Config.FOVEA_OUTPUT_SIZE,
                                      crop_size=Config.FOVEA_CROP_SIZE)
                foveated_image = torch.from_numpy(blurred.astype(np.float32)/255.0)\
                                    .permute(2,0,1).unsqueeze(0).to(Config.DEVICE)
                
                if stop_action.item() == 1:
                    if batch_idx < 2: 
                        break   

            # average over all steps
            total_rl_loss = total_rl_loss * Config.RL_LOSS_WEIGHT
            total_rec_loss = total_rec_loss

            # Combine losses
            total_loss = total_rl_loss + total_rec_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update counters and maybe checkpoint every 5000 images
            imgs_seen += images.size(0)
            if imgs_seen >= next_save_at:
                ckpt_path = os.path.join(run_dir, f"model_images_{imgs_seen}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at {imgs_seen} images -> {ckpt_path}")
                next_save_at += 5000

            global_step = epoch * len(train_loader) + batch_idx
            # log reconstruction and RL losses separately
            writer.add_scalar("Loss/rec_loss", total_rec_loss.item(), global_step)
            writer.add_scalar("Loss/rl_loss", total_rl_loss.item(), global_step)
            writer.add_scalar("Episode/length", step, global_step)

            # Occasionally log image (every 10 batches)
            if batch_idx % 10 == 0:
                # Create a grid with the original image and the final reconstruction
                grid = vutils.make_grid(torch.cat([image, reconstruction], dim=0), nrow=2, normalize=True)
                writer.add_image("Episode/original_vs_reconstruction", grid, global_step)
                print(f"Epoch {epoch} Batch {batch_idx}: rec_loss={total_loss.item():.6f}")

            clear_memory()
        model_path = os.path.join("gaze_control_model_local.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch} complete. Model saved to {model_path}")
    writer.close()

class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        
        # Get all image files recursively from the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        for ext in image_extensions:
            # Search recursively with **/ pattern
            pattern = os.path.join(data_dir, '**', ext)
            self.image_paths.extend(glob.glob(pattern, recursive=True))
            # Also search for uppercase extensions
            pattern = os.path.join(data_dir, '**', ext.upper())
            self.image_paths.extend(glob.glob(pattern, recursive=True))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Filter out invalid images
        valid_paths = []
        for path in self.image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()  # Verify it's a valid image
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping invalid image {path}: {e}")
        
        self.image_paths = valid_paths
        print(f"Found {len(self.image_paths)} valid images in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # Return dummy label since you don't need classes
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            black_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                black_img = self.transform(black_img)
            return black_img, 0

if __name__ == "__main__":
    train()
