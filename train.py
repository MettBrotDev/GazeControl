import os
# silence TF/XLA plugin re‐registration messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # suppress TF logs
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import datetime
import numpy as np
from foveal_blur import foveal_blur
import torch.distributions as D

from models import GazeControlModel
from config import Config          

def clear_memory():
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

def train():
    # timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"gaze_control_{Config.DATA_SOURCE}_{timestamp}"
    run_dir = os.path.join("runs", log_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)  # Initialize tensorboard writer
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
        train_dataset = datasets.ImageFolder(root=Config.LOCAL_DATA_DIR,
                                             transform=transform_img)

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)

    # Initialize model, loss and optimizer
    # Current Task: Full reconstruction.
    model = GazeControlModel(encoder_output_size=Config.ENCODER_OUTPUT_SIZE,
                             hidden_size=Config.HIDDEN_SIZE,
                             img_size=Config.IMG_SIZE,
                             fovea_size=Config.FOVEA_OUTPUT_SIZE).to(Config.DEVICE)
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

            # Initialize recurrent state
            h = torch.zeros(1, Config.HIDDEN_SIZE, device=Config.DEVICE)
            c = torch.zeros(1, Config.HIDDEN_SIZE, device=Config.DEVICE)

            reconstruction = None

            # Perform a fixed number of gazemovements 
            for step in range(Config.MAX_STEPS):
                if step == 0:
                    # No previous rec_error and initialize next_value for TD target
                    prev_rec_error = torch.tensor(0.0, device=Config.DEVICE)
                # rl step
                reconstruction, (gaze_delta, stop_action), _, (h_new,c_new) = \
                    model.sample_action(foveated_image, h, c, gaze)

                # compute new reconstruction error
                rec_error = rec_loss_fn(reconstruction, image) * Config.RECONSTRUCTION_WEIGHT

                rec_loss = rec_error.clone()

                # reward = how much error dropped minus cost of an extra glimpse
                # !!!!!!! WATCH OUT THIS IS ONLY FOR THIS TASK, if the Reward is caluclated differently its not useful anymore
                improvement = (prev_rec_error - rec_error)
                reward = improvement.detach() * Config.REWARD_SCALE - Config.EFFICIENCY_PENALTY
                if step == 0:
                    # Dont scale the first one since that one is pretty high by design
                    reward = improvement.detach() - Config.EFFICIENCY_PENALTY

                prev_rec_error = rec_error.detach()

                # predict next state value for TD target
                _, _, _, next_value = model.agent.forward(h_new.detach())
                td_target = reward + Config.GAMMA * next_value.detach()

                # compute value prediction & loss
                _, _, _, value_pred = model.agent.forward(h)

                # 1) Critic loss
                value_loss = nn.functional.mse_loss(value_pred, td_target.detach())
                # 2) Actor (policy) loss
                gaze_mean, gaze_std, stop_logit, _ = model.agent.forward(h)
                gaze_dist = D.Normal(gaze_mean, gaze_std)
                stop_prob = torch.sigmoid(stop_logit)
                stop_dist = D.Bernoulli(stop_prob)

                logp_gaze = gaze_dist.log_prob(gaze_delta).sum(-1)      # sum dims dx,dy
                logp_stop = stop_dist.log_prob(stop_action)
                advantage = (td_target - value_pred).detach()
                policy_loss = -(logp_gaze + logp_stop) * advantage

                # Entropy bonus for exploration
                ent_gaze = gaze_dist.entropy().sum(-1)  # sum dims dx,dy
                ent_stop = stop_dist.entropy()
                entropy = (ent_gaze + ent_stop).mean()

                # 3) Combined loss
                rl_loss = Config.VALUE_WEIGHT * value_loss \
                        + Config.POLICY_WEIGHT * policy_loss.mean() \
                        - Config.ENTROPY_WEIGHT * entropy
                
                #print(f'[DEBUG] Reconstruction Loss: {rec_error.item():.4f}, Value Loss: {value_loss.item():.4f}, Policy Loss: {policy_loss.mean().item():.4f}, Entropy: {entropy.item():.4f}')

                total_rl_loss = total_rl_loss + rl_loss
                total_rec_loss = total_rec_loss + rec_loss * np.power(Config.STEP_RECONSTRUCTION_DISCOUNT, (step + 1) / Config.MAX_STEPS)

                # update state & foveated_image (detach to avoid in‐place errors)
                h, c = h_new, c_new
                # clamp and apply gaze_delta
                max_m = torch.tensor([Config.MAX_MOVE,
                                      Config.MAX_MOVE],
                                     device=Config.DEVICE, dtype=gaze_delta.dtype)
                gaze_delta = gaze_delta * max_m
                # Ensure gaze_delta is within bounds
                gaze = (gaze + gaze_delta).detach()
                gaze = torch.clamp(gaze, 0, 1)
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
                    pass
                    #break         # Currently disabled due to issues during training

            # average over all steps
            total_rl_loss = total_rl_loss 
            total_rec_loss = total_rec_loss

            # Combine losses
            total_loss = total_rl_loss + total_rec_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

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

if __name__ == "__main__":
    train()
