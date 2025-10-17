import os
# silence TF/XLA plugin re‐registration messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
try:
    import wandb
except Exception:
    wandb = None
import datetime
import numpy as np
import glob
from PIL import Image
import math
import argparse
import importlib
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from models import GazeControlModel, Agent, PerceptualLoss
from config import Config          

def clear_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def crop_patch(image, gaze, crop_size, resize_to=None):
    """Crop a patch around the gaze from a batched image tensor.
    image: (B,3,H,W) in [0,1]
    gaze: (B,2) normalized [0,1]
    crop_size: (h,w) desired crop before optional resize
    resize_to: (h,w) to resize with bilinear if provided
    Returns (B,3,h,w)
    """
    B, C, H, W = image.shape
    ch, cw = crop_size
    crops = []
    for i in range(B):
        cx = int(gaze[i, 0].item() * (W - 1))
        cy = int(gaze[i, 1].item() * (H - 1))
        x0 = cx - cw // 2
        y0 = cy - ch // 2
        x1 = x0 + cw
        y1 = y0 + ch

        # Clamp source region to image bounds
        x0_src = max(0, min(W, x0))
        y0_src = max(0, min(H, y0))
        x1_src = max(0, min(W, x1))
        y1_src = max(0, min(H, y1))

        crop = image[i:i+1, :, y0_src:y1_src, x0_src:x1_src]

        # Compute non-negative padding to reach target crop size
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            # zero pad out-of-bounds regions for more intuitive behavior
            crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

        # Ensure exact target crop size (in case of rounding)
        if crop.shape[-2:] != (ch, cw):
            crop = F.interpolate(crop, size=(ch, cw), mode='bilinear', align_corners=False)

        crops.append(crop)

    crops = torch.cat(crops, dim=0)
    if resize_to is not None and (resize_to[0] != ch or resize_to[1] != cw):
        crops = F.interpolate(crops, size=resize_to, mode='bilinear', align_corners=False)
    return crops

def _make_gaussian_mask(H, W, gaze_xy, sigma_frac=0.25, device="cpu"):
    """Create soft Gaussian masks centered at gaze_xy (normalized [0,1]).
    gaze_xy: (B,2)
    Returns (B,1,H,W).
    """
    if gaze_xy.dim() == 1:
        gaze_xy = gaze_xy.view(1, 2)
    B = gaze_xy.size(0)
    gx = gaze_xy[:, 0].clamp(0, 1) * (W - 1)  # (B,)
    gy = gaze_xy[:, 1].clamp(0, 1) * (H - 1)  # (B,)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xx = xx.unsqueeze(0).float()  # (1,H,W)
    yy = yy.unsqueeze(0).float()  # (1,H,W)
    dx2 = (xx - gx.view(B, 1, 1)) ** 2
    dy2 = (yy - gy.view(B, 1, 1)) ** 2
    sigma = sigma_frac * float(min(H, W))
    sigma2 = max(1e-6, sigma * sigma)
    mask = torch.exp(-(dx2 + dy2) / (2.0 * sigma2))  # (B,H,W)
    return mask.unsqueeze(1)  # (B,1,H,W)


def eight_dir_deltas(max_move: float, device: str):
    """Return 8 unit directions scaled by max_move in order: E, NE, N, NW, W, SW, S, SE.
    Shape: (8, 2)
    """
    dirs = torch.tensor([
        [1.0, 0.0],   # E
        [1.0, 1.0],   # NE
        [0.0, 1.0],   # N
        [-1.0, 1.0],  # NW
        [-1.0, 0.0],  # W
        [-1.0, -1.0], # SW
        [0.0, -1.0],  # S
        [1.0, -1.0],  # SE
    ], device=device)
    # normalize diagonals to unit length, then scale
    dirs = dirs / torch.clamp(dirs.norm(dim=1, keepdim=True), min=1e-6)
    return dirs * max_move


def train(use_pretrained_decoder=True, load_full_model=False, no_rl=False, wb=None):
    # timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"gaze_control_{Config.DATA_SOURCE}_rl_{timestamp}"
    run_dir = os.path.join("runs", log_name)
    os.makedirs(run_dir, exist_ok=True)
    # TensorBoard disabled; use W&B if provided
    
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
    model = GazeControlModel(encoder_output_size=Config.ENCODER_OUTPUT_SIZE,
                             state_size=Config.HIDDEN_SIZE,
                             img_size=Config.IMG_SIZE,
                             fovea_size=Config.FOVEA_OUTPUT_SIZE,
                             pos_encoding_dim=Config.POS_ENCODING_DIM,
                             lstm_layers=Config.LSTM_LAYERS,
                             decoder_latent_ch=Config.DECODER_LATENT_CH,
                             k_scales=getattr(Config, 'K_SCALES', 3),
                             fuse_to_dim=getattr(Config, 'FUSION_TO_DIM', None),
                             fusion_hidden_mul=getattr(Config, 'FUSION_HIDDEN_MUL', 2.0),
                             encoder_c1=getattr(Config, 'ENCODER_C1', None),
                             encoder_c2=getattr(Config, 'ENCODER_C2', None)).to(Config.DEVICE)

    # Load a full pretrained model checkpoint (e.g., random-move baseline) if configured
    ckpt_path = getattr(Config, 'PRETRAINED_MODEL_PATH', "")
    if load_full_model and ckpt_path:
        if os.path.exists(ckpt_path):
            print(f"Loading full model checkpoint from {ckpt_path}")
            raw = torch.load(ckpt_path, map_location=Config.DEVICE)
            # Handle common checkpoint formats
            if isinstance(raw, dict) and 'state_dict' in raw:
                state = raw['state_dict']
            elif isinstance(raw, dict) and 'model_state_dict' in raw:
                state = raw['model_state_dict']
            else:
                state = raw
            try:
                # Log match stats
                model_state = model.state_dict()
                matched = [k for k in state.keys() if k in model_state and getattr(model_state[k], 'shape', None) == getattr(state[k], 'shape', None)]
                res = model.load_state_dict(state, strict=False)
                print(f"Full checkpoint loaded: matched={len(matched)} missing={len(getattr(res, 'missing_keys', []))} unexpected={len(getattr(res, 'unexpected_keys', []))}")
            except Exception as e:
                print(f"Warning: failed to load full checkpoint ({e}).")
        else:
            print(f"Warning: PRETRAINED_MODEL_PATH set but not found: {ckpt_path}")

    # Load pretrained decoder if available and requested
    if use_pretrained_decoder and os.path.exists(Config.PRETRAINED_DECODER_PATH):
        print(f"Loading pretrained decoder from {Config.PRETRAINED_DECODER_PATH}")
        decoder_state = torch.load(Config.PRETRAINED_DECODER_PATH, map_location=Config.DEVICE)
        # Try strict load; if channel mismatch (e.g., latent_ch changed), load best effort
        try:
            model.decoder.load_state_dict(decoder_state, strict=True)
        except Exception as e:
            print(f"Strict load failed ({e}); attempting non-strict load")
            model.decoder.load_state_dict(decoder_state, strict=False)
        print("Pretrained decoder loaded successfully!")
        
        # Optionally freeze decoder for first few epochs
        if hasattr(Config, 'FREEZE_DECODER_EPOCHS') and Config.FREEZE_DECODER_EPOCHS > 0:
            print(f"Freezing decoder for first {Config.FREEZE_DECODER_EPOCHS} epoch(s)")
            for param in model.decoder.parameters():
                param.requires_grad = False
    elif use_pretrained_decoder:
        print(f"No pretrained decoder found at {Config.PRETRAINED_DECODER_PATH}, training from scratch")
    else:
        print("Decoder pretrain disabled; continuing with current model weights")

    # No Gaussian std for discrete policy

    # Build separate optimizers for policy/value heads and backbone
    agent = Agent(state_size=Config.HIDDEN_SIZE, pos_encoding_dim=Config.POS_ENCODING_DIM).to(Config.DEVICE)
    policy_params = list(agent.parameters())
    backbone_params = list(model.parameters())

    opt_policy = None
    if not no_rl:
        opt_policy = torch.optim.AdamW(policy_params, lr=getattr(Config, 'RL_POLICY_LR', Config.LEARNING_RATE), weight_decay=Config.WEIGHT_DECAY)
    else:
        # Freeze policy/value heads when RL is disabled
        for p in policy_params:
            p.requires_grad = False
    opt_backbone = torch.optim.AdamW(backbone_params, lr=getattr(Config, 'RL_BACKBONE_LR', Config.LEARNING_RATE), weight_decay=Config.WEIGHT_DECAY)  # RL backbone lr not in config rn

    l1_loss = torch.nn.L1Loss()
    # Store loss weights as local variables for consistency with pretrain script
    l1_weight = float(getattr(Config, 'L1_WEIGHT', 1.0))
    # Optional perceptual loss on final reconstruction (disabled if weight <= 0)
    perc_weight = float(getattr(Config, 'PERC_WEIGHT', 0.0))
    criterion_perc = PerceptualLoss().to(Config.DEVICE) if perc_weight > 0.0 else None
    # MS-SSIM loss
    ssim_weight = float(getattr(Config, 'SSIM_WEIGHT', 0.0))
    use_ssim = ssim_weight > 0.0
    # Foreground mask for L1 loss
    use_fg_mask = bool(getattr(Config, 'USE_FG_MASK', False))
    fg_thresh = float(getattr(Config, 'FG_THRESH', 0.1))
    bg_weight = float(getattr(Config, 'BG_WEIGHT', 0.1))

    # RL is always fully attached; no detach/ramp schedule

    # Optionally freeze backbone for initial RL-only epochs
    rl_only_epochs = int(getattr(Config, 'RL_ONLY_EPOCHS', 0))
    if no_rl:
        rl_only_epochs = 0  # don't freeze backbone when RL is disabled
    if rl_only_epochs > 0 and not no_rl:
        print(f"RL-first enabled: freezing backbone for first {rl_only_epochs} epoch(s)")
        for p in model.parameters():
            p.requires_grad = False
    model.train()
    global_step = 0
    episodes_seen = 0
    for epoch in range(Config.EPOCHS):
        # Transition out of RL-only phase: unfreeze backbone at epoch boundary
        if (not no_rl) and rl_only_epochs > 0 and epoch == rl_only_epochs:
                print(f"Unfreezing backbone after RL-only phase at epoch {epoch}")
                for p in model.parameters():
                    p.requires_grad = True
                # If decoder freeze policy is configured, re-freeze decoder for remaining initial epochs
                if hasattr(Config, 'FREEZE_DECODER_EPOCHS') and Config.FREEZE_DECODER_EPOCHS > 0:
                    print(f"Applying decoder freeze for {Config.FREEZE_DECODER_EPOCHS} epoch(s) after RL-only phase")
                    for param in model.decoder.parameters():
                        param.requires_grad = False

        # Unfreeze decoder after FREEZE_DECODER_EPOCHS (counts from when decoder was frozen)
        if (hasattr(Config, 'FREEZE_DECODER_EPOCHS') and 
            Config.FREEZE_DECODER_EPOCHS > 0 and 
            epoch == rl_only_epochs + Config.FREEZE_DECODER_EPOCHS):
            print(f"Unfreezing decoder at epoch {epoch}")
            for param in model.decoder.parameters():
                param.requires_grad = True
                
        for batch_idx, (images, _) in enumerate(train_loader):
            total_rec_loss = torch.tensor(0.0, device=Config.DEVICE)
            image = images.to(Config.DEVICE)
            
            # Start with random initial gaze position
            num_steps = Config.MAX_STEPS
            B = image.size(0)
            gaze = torch.rand(B, 2, device=Config.DEVICE) * 0.4 + 0.3  # Start in central 40% area

            # Initialize LSTM memory
            state = model.init_memory(B, Config.DEVICE)
            reconstruction = None

            attach_alpha = 1.0

            # No coupling schedule to log anymore

            # RL rollouts storage
            logprobs = []
            values = []
            entropies = []
            rewards = []
            prev_full_err = None
            # record gaze positions per step (pre-action)
            gaze_path = []

            # Accumulate per-step losses (optionally masked/local)
            for step in range(num_steps):
                # record current gaze before taking action
                gaze_path.append(gaze[0].detach().cpu())
                # Multi-scale glimpse: k scales with sizes base*(2**i), all resized to FOVEA_OUTPUT_SIZE
                k_scales = int(getattr(Config, 'K_SCALES', 3))
                base_h, base_w = Config.FOVEA_CROP_SIZE
                patches = []
                for i in range(k_scales):
                    scale = 2 ** i
                    size = (base_h * scale, base_w * scale)
                    patches.append(crop_patch(image, gaze, size, resize_to=Config.FOVEA_OUTPUT_SIZE))
                reconstruction, state = model(patches, state, gaze)

                if getattr(Config, "USE_MASKED_STEP_LOSS", False):
                    # Soft local mask around gaze; compute losses only where observed
                    H, W = Config.IMG_SIZE
                    sigma_base = Config.FOVEA_OUTPUT_SIZE[0] / max(H, W)
                    sigma_frac = sigma_base * float(getattr(Config, "STEP_MASK_SIGMA_SCALE", 0.35))
                    mask = _make_gaussian_mask(H, W, gaze, sigma_frac=sigma_frac, device=Config.DEVICE)  # (B,1,H,W)
                    # Broadcast to channels and compute per-sample masked loss, then mean over batch
                    mask3 = mask.expand(-1, 3, H, W)
                    per_sample_l1 = (((reconstruction - image).abs() * mask3).flatten(1).sum(dim=1) /
                                     (mask3.flatten(1).sum(dim=1) + 1e-6))
                    l1_m = per_sample_l1.mean()
                else:
                    l1_m = 0
                
                # Weight increases linearly between STEP_LOSS_MIN and STEP_LOSS_MAX across steps
                step_frac = (step + 1) / max(1, num_steps)
                min_w = getattr(Config, "STEP_LOSS_MIN", 0.02)
                max_w = getattr(Config, "STEP_LOSS_MAX", 0.2)
                weight = min_w + (max_w - min_w) * step_frac

                # Compute L1 loss with optional foreground masking
                if use_fg_mask:
                    # Compute luminance for foreground detection
                    img_y = 0.2989 * image[:,0:1] + 0.5870 * image[:,1:2] + 0.1140 * image[:,2:3]
                    recon_y = 0.2989 * reconstruction[:,0:1] + 0.5870 * reconstruction[:,1:2] + 0.1140 * reconstruction[:,2:3]
                    # Create weight mask: foreground=1.0, background=bg_weight
                    w = torch.where(img_y >= fg_thresh, torch.ones_like(img_y), torch.full_like(img_y, bg_weight))
                    # Apply weight to L1 error
                    l1_step = (w * (recon_y - img_y).abs()).mean() * weight + l1_m
                else:
                    l1_step = l1_loss(reconstruction, image) * weight + l1_m

                # Compute MS-SSIM loss for this step if enabled
                if use_ssim:
                    mssim_step_val = ms_ssim(reconstruction, image, data_range=1.0, size_average=True)
                    ssim_step = (1.0 - mssim_step_val) * weight
                else:
                    ssim_step = 0.0

                rec_error = l1_weight * l1_step + (ssim_weight * ssim_step if use_ssim else 0.0)

                total_rec_loss = total_rec_loss + rec_error

                # ----- RL: compute reward per-sample from improvement in FULL reconstruction loss -----
                with torch.no_grad():
                    # Per-sample full L1 loss to match per-sample actions/values
                    full_l1_per = F.l1_loss(reconstruction, image, reduction='none').mean(dim=(1,2,3))    # (B,)
                    curr_full_err = l1_weight * full_l1_per     # (B,)
                    if prev_full_err is None:
                        # first step: set baseline only
                        prev_full_err = curr_full_err.detach()
                    else:
                        r_t = (prev_full_err - curr_full_err)  # (B,)
                        rewards.append(r_t)
                        prev_full_err = curr_full_err.detach()

                # Choose action: RL policy or uniform random over 8 directions
                if not no_rl:
                    # Policy/value from current hidden state with gaze context (top layer)
                    h_t = state[0][-1]  # (B,H)
                    # Always fully attached now
                    logits, v_t = agent.policy_value(h_t, gaze)  # logits (B,8), value (B,)
                    cat = Categorical(logits=logits)
                    action_idx = cat.sample()               # (B,)
                    logprob = cat.log_prob(action_idx)      # (B,)
                    entropy = cat.entropy()                 # (B,)
                else:
                    action_idx = torch.randint(low=0, high=8, size=(gaze.shape[0],), device=Config.DEVICE)
                # Map discrete action index to delta
                deltas = eight_dir_deltas(Config.MAX_MOVE, device=Config.DEVICE)
                delta = deltas[action_idx]

                # Apply action for next step
                gaze = torch.clamp(gaze + delta, 0.0, 1.0)
                # Keep gaze within central bounds if enabled
                if getattr(Config, 'USE_GAZE_BOUNDS', False):
                    frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
                    lo, hi = frac, 1.0 - frac
                    gaze = gaze.clamp(min=lo, max=hi)

                if step > 0 and not no_rl:
                    # again skip first step since no action was taken yet
                    logprobs.append(logprob)
                    values.append(v_t)
                    entropies.append(entropy)

            # Final reconstruction from final LSTM state with larger weight
            final_recon = model.decode_from_state(state)
            
            # Compute L1 loss with optional foreground masking
            if use_fg_mask:
                img_y = 0.2989 * image[:,0:1] + 0.5870 * image[:,1:2] + 0.1140 * image[:,2:3]
                final_y = 0.2989 * final_recon[:,0:1] + 0.5870 * final_recon[:,1:2] + 0.1140 * final_recon[:,2:3]
                w = torch.where(img_y >= fg_thresh, torch.ones_like(img_y), torch.full_like(img_y, bg_weight))
                final_l1 = (w * (final_y - img_y).abs()).mean()
            else:
                final_l1 = l1_loss(final_recon, image)
            
            # Optional perceptual loss
            final_perc = None
            if criterion_perc is not None:
                final_perc = criterion_perc(final_recon, image)
            
            # Optional MS-SSIM loss
            final_ssim = None
            if use_ssim:
                # MS-SSIM expects inputs in [0,1] range
                mssim_val = ms_ssim(final_recon, image, data_range=1.0, size_average=True)
                final_ssim = 1.0 - mssim_val
            
            final_mult = getattr(Config, "FINAL_LOSS_MULT", 8.0)
            final_loss = l1_weight * final_l1 * final_mult
            if final_perc is not None and perc_weight > 0.0:
                final_loss = final_loss + perc_weight * final_perc * final_mult
            if final_ssim is not None and ssim_weight > 0.0:
                final_loss = final_loss + ssim_weight * final_ssim * final_mult

            total_loss = total_rec_loss + final_loss

            # ----- RL loss (A2C with GAE) -----
            rl_loss = torch.tensor(0.0, device=Config.DEVICE)
            if (not no_rl) and len(rewards) > 0:
                gamma = float(getattr(Config, 'RL_GAMMA', 0.95))
                lam = float(getattr(Config, 'RL_LAMBDA', 0.95))
                rewards_t_raw = torch.stack(rewards)  # (T,B)
                scale = float(getattr(Config, 'RL_REWARD_SCALE', 1.0))
                rewards_t = rewards_t_raw * scale  # (T, B=1)
                logprobs_t = torch.stack(logprobs)  # (T, B)
                values_t = torch.stack(values)      # (T, B)
                entropies_t = torch.stack(entropies)  # (T, B)

                with torch.no_grad():
                    # bootstrap with critic using final hidden state and current gaze
                    h_T = state[0][-1]
                    next_value = agent.policy_value(h_T, gaze)[1]  # (B,)
                    T = rewards_t.size(0)
                    advantages = torch.zeros_like(values_t)
                    gae = torch.zeros_like(next_value)
                    for t in reversed(range(T)):
                        v_t = values_t[t]
                        v_tp1 = next_value if t == T - 1 else values_t[t + 1]
                        delta = rewards_t[t] + gamma * v_tp1 - v_t
                        gae = delta + gamma * lam * gae
                        advantages[t] = gae
                    returns = advantages + values_t

                # Optionally normalize advantages
                if bool(getattr(Config, 'RL_NORM_ADV', False)):
                    adv_mean = advantages.mean()
                    adv_std = advantages.std().clamp_min(1e-6)
                    advantages = (advantages - adv_mean) / adv_std

                policy_loss = -(logprobs_t * advantages.detach()).mean()
                value_loss = F.mse_loss(values_t, returns)
                entropy_loss = -entropies_t.mean()

                value_coef = float(getattr(Config, 'RL_VALUE_COEF', 0.5))
                entropy_coef = float(getattr(Config, 'RL_ENTROPY_COEF', 0.01))
                rl_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                # Log RL metrics via W&B
                if wb is not None:
                    wandb.log({
                        "RL/policy_loss": float(policy_loss.item()),
                        "RL/value_loss": float(value_loss.item()),
                        "RL/entropy": float((-entropy_loss).item()),
                        "RL/mean_reward_raw": float(rewards_t_raw.mean().item()),
                        "RL/mean_reward_scaled": float(rewards_t.mean().item()),
                        "RL/mean_advantage": float(advantages.mean().item()),
                    }, step=global_step)

            total_loss = total_loss + float(getattr(Config, 'RL_LOSS_WEIGHT', 1.0)) * rl_loss

            # Backpropagation
            if opt_policy is not None:
                opt_policy.zero_grad()
            if epoch >= rl_only_epochs:
                opt_backbone.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability (clip both backbone and agent)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
            if opt_policy is not None:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=Config.GRAD_CLIP_NORM)

            # Step optimizers
            if opt_policy is not None:
                opt_policy.step()
            if epoch >= rl_only_epochs:
                opt_backbone.step()

            # Update counters and maybe checkpoint every 5000 images
            imgs_seen += images.size(0)
            if imgs_seen >= next_save_at:
                ckpt_path = os.path.join(run_dir, f"model_images_{imgs_seen}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at {imgs_seen} images -> {ckpt_path}")
                next_save_at += 50000

            # log reconstruction loss via W&B
            rec_loss_unscaled = l1_weight * final_l1
            if wb is not None:
                logs = {
                    "loss/rec_l1": float(rec_loss_unscaled.item()),
                    "loss/total": float(total_loss.item()),
                    "episode/steps": int(Config.MAX_STEPS),
                }
                if final_perc is not None:
                    logs["loss/perc"] = float(final_perc.item())
                if final_ssim is not None:
                    logs["loss/ssim"] = float(final_ssim.item())
                wandb.log(logs, step=global_step)

            # Occasionally log a synchronized figure: Original+GazePath vs Reconstruction (every 10 batches)
            if batch_idx % 10 == 0:
                try:
                    H, W = Config.IMG_SIZE
                    # Prepare numpy images
                    orig_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
                    recon_np = reconstruction[0].detach().cpu().permute(1, 2, 0).numpy()
                    # Build combined figure
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    # Left: Original with gaze path
                    axs[0].imshow(orig_np)
                    if len(gaze_path) > 0:
                        pts = [(int(p[0].item() * (W - 1)), int(p[1].item() * (H - 1))) for p in gaze_path]
                        xs, ys = zip(*pts)
                        axs[0].scatter(xs, ys, c='r', s=40, marker='x', label='gaze')
                        axs[0].plot(xs, ys, c='yellow', linewidth=1, alpha=0.8)
                    axs[0].set_title('Original + gaze path')
                    axs[0].axis('off')
                    # Right: Final step reconstruction
                    axs[1].imshow(recon_np)
                    axs[1].set_title('Reconstruction (final step)')
                    axs[1].axis('off')
                    plt.tight_layout()
                    if wb is not None:
                        try:
                            wandb.log({"train/original_gaze_vs_recon": wandb.Image(fig)}, step=global_step)
                        except Exception:
                            pass
                    plt.close(fig)
                except Exception:
                    pass
                print(f"Epoch {epoch} Batch {batch_idx}: rec_loss={rec_loss_unscaled.item():.6f} total={total_loss.item():.6f}")

            global_step += 1
            episodes_seen += 1
            clear_memory()
        model_path = os.path.join("gaze_control_model_local.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch} complete. Model saved to {model_path}")
    # No TensorBoard writer to close

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
    parser = argparse.ArgumentParser(description='Train GazeControl model')
    parser.add_argument('--config', default='config', help="config module name (e.g. 'config', 'configLarge', 'configXL')")
    parser.add_argument('--use-pretrained', action='store_true', 
                        help='Enable loading pretrained decoder weights (Config.PRETRAINED_DECODER_PATH)')
    parser.add_argument('--load-full', action='store_true', 
                        help='Load full model from Config.PRETRAINED_MODEL_PATH before training')
    parser.add_argument('--no-rl', action='store_true', 
                        help='Disable RL actor/critic; sample random actions from the discrete action space')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='GazeControl', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    # Dynamically load config module and rebind Config used in this module
    if getattr(args, 'config', None) and args.config != 'config':
        try:
            mod = importlib.import_module(args.config)
            Config = mod.Config  # rebind global used above
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Failed to load config module '{args.config}', falling back to default. Error: {e}")

    # Brief summary of key model dims for sanity
    try:
        print(f"Config: IMG={Config.IMG_SIZE}, FOVEA={Config.FOVEA_OUTPUT_SIZE}, HIDDEN={Config.HIDDEN_SIZE}, POS_DIM={Config.POS_ENCODING_DIM}, LAYERS={Config.LSTM_LAYERS}, FUSION_TO_DIM={getattr(Config,'FUSION_TO_DIM',None)}, DEC_CH={Config.DECODER_LATENT_CH}, BATCH={Config.BATCH_SIZE}")
    except Exception:
        pass

    wb_run = None
    if args.wandb and wandb is not None:
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "img_size": Config.IMG_SIZE,
                "batch_size": Config.BATCH_SIZE,
                "lr": Config.LEARNING_RATE,
                "device": Config.DEVICE,
                "data_source": Config.DATA_SOURCE,
                "k_scales": getattr(Config, 'K_SCALES', 3),
                "fovea": Config.FOVEA_OUTPUT_SIZE,
            }
        )

    if args.no_rl:
        print("RL disabled (--no_rl): using uniform random actions from the discrete space.")
    try:
        train(use_pretrained_decoder=args.use_pretrained, load_full_model=args.load_full, no_rl=args.no_rl, wb=wb_run)
    finally:
        if wb_run is not None:
            wandb.finish()
