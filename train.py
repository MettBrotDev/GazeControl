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
import json
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from models import GazeControlModel, Agent, PerceptualLoss, GradientDifferenceLoss
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


class MazeDataset(Dataset):
    """Maze dataset that reads images and labels from generated metadata JSON."""
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__()
        self.root = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'imgs', split)
        self.meta_path = os.path.join(root_dir, f'{split}_metadata.json')
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        # Keep full entries so we can access start/end if present
        self.samples = self.meta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        fname = entry['filename']
        label = int(entry['label'])
        p = os.path.join(self.img_dir, fname)
        
        # Load image
        img_pil = Image.open(p).convert('RGB')

        # Just take gridsize like this for now
        self.grid_size = Config.IMG_SIZE[0] // 4

        # Compute start gaze from metadata grid coordinate
        sx = sy = None
        if isinstance(entry.get('start'), dict):
            try:
                sx = int(entry['start'].get('x'))
                sy = int(entry['start'].get('y'))
            except Exception:
                sx = sy = None
        if sx is not None and sy is not None:
            nx = (float(sx) + 0.5) / float(self.grid_size)
            ny = (float(sy) + 0.5) / float(self.grid_size)
            start_xy = torch.tensor([nx, ny], dtype=torch.float32)
        else:
            # Fallback: center if metadata is missing
            start_xy = torch.tensor([0.5, 0.5], dtype=torch.float32)

        # Apply transforms to image
        img = self.transform(img_pil) if self.transform else img_pil

        return img, torch.tensor(label, dtype=torch.long), start_xy


class SnakesDataset(Dataset):
    """Snakes2 dataset loader.
    Expects `root_dir` containing `imgs/<batch_id>/sample_*.png` and `metadata/<batch_id>.npy`.
    Uses labels from metadata and spawns gaze at the image center (no spawnpoint available).
    Splits train/val deterministically with a fixed fraction.
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None, val_frac: float = 0.1, seed: int = 1337):
        super().__init__()
        self.root = root_dir
        self.split = split
        self.transform = transform
        self.val_frac = float(val_frac)
        self.seed = int(seed)

        # Load all metadata rows from npy files
        meta_dir = os.path.join(root_dir, 'metadata')
        rows = []
        if os.path.isdir(meta_dir):
            for fn in sorted(os.listdir(meta_dir)):
                if not fn.lower().endswith('.npy'):
                    continue
                try:
                    arr = np.load(os.path.join(meta_dir, fn), allow_pickle=True)
                    # rows: [subpath, filename, nimg, label, ...]
                    for r in arr:
                        try:
                            subpath, filename, _, label = r[0], r[1], r[2], int(r[3])
                            img_path = os.path.join(root_dir, subpath, filename)
                            rows.append((img_path, label))
                        except Exception:
                            continue
                except Exception:
                    continue
        # Fallback: scan imgs/ if metadata missing (label=0)
        if not rows:
            img_root = os.path.join(root_dir, 'imgs')
            for dp, _, files in os.walk(img_root):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        rows.append((os.path.join(dp, f), 0))

        # Filter to existing files
        rows = [(p, lbl) for (p, lbl) in rows if os.path.exists(p)]
        # Deterministic split
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        split_at = int(round((1.0 - self.val_frac) * len(rows)))
        if split == 'train':
            take = idx[:split_at]
        else:
            take = idx[split_at:]
        self.samples = [rows[i] for i in take]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        try:
            img_pil = Image.open(p).convert('RGB')
        except Exception:
            img_pil = Image.new('RGB', (Config.IMG_SIZE[1], Config.IMG_SIZE[0]), color='black')
        img = self.transform(img_pil) if self.transform else img_pil
        # Spawn in the center (normalized)
        start_xy = torch.tensor([0.5, 0.5], dtype=torch.float32)
        return img, torch.tensor(int(label), dtype=torch.long), start_xy


def train(
    use_pretrained_decoder=True,
    load_full_model=False,
    no_rl=False,
    wb=None,
    recon_warmup_epochs: int = 0,
    no_cls_warmup: bool = False,
):
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
    # NOTE: Decoder was pretrained on [-1,1] range (Tanh output), so we must match that range
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] → [-1,1]
    ])
    transform_mnist = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] → [-1,1]
    ])

    # Choose dataset per config
    data_src = getattr(Config, 'DATA_SOURCE', 'maze').lower()
    if data_src == 'snakes':
        snakes_root = getattr(Config, 'SNAKES_ROOT', './Data/Snakes128_tight')
        if not os.path.exists(snakes_root):
            raise FileNotFoundError(f"Snakes dataset not found at {snakes_root}. Generate with Datasets/Pathfinder/generate_snakes2_tight128.py")
        val_frac = float(getattr(Config, 'SNAKES_VAL_FRAC', 0.1))
        train_dataset = SnakesDataset(snakes_root, split='train', transform=transform_img, val_frac=val_frac)
        val_dataset = SnakesDataset(snakes_root, split='val', transform=transform_img, val_frac=val_frac)
    else:
        # Maze dataset and its JSON metadata
        maze_root = getattr(Config, 'MAZE_ROOT', os.path.join('./Data', 'Maze'))
        if not os.path.exists(os.path.join(maze_root, 'imgs', 'train')):
            raise FileNotFoundError(f"Maze dataset not found at {maze_root}. Set Config.MAZE_ROOT to your dataset root or generate it with Datasets/generate_maze.py")
        train_dataset = MazeDataset(maze_root, split='train', transform=transform_img)
        val_dataset = MazeDataset(maze_root, split='val', transform=transform_img)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

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
    agent = Agent(state_size=Config.HIDDEN_SIZE, pos_encoding_dim=Config.POS_ENCODING_DIM,
                  stop_init_bias=float(getattr(Config, 'RL_STOP_INIT_BIAS', -8.0))).to(Config.DEVICE)
    # If a checkpoint was requested via --load-full and it contains an agent_state_dict, load it now
    if load_full_model and ckpt_path and os.path.exists(ckpt_path):
        try:
            raw2 = torch.load(ckpt_path, map_location=Config.DEVICE)
            if isinstance(raw2, dict) and 'agent_state_dict' in raw2:
                agent.load_state_dict(raw2['agent_state_dict'], strict=False)
                print("Loaded agent_state_dict from checkpoint")
        except Exception as e:
            print(f"Warning: failed to load agent_state_dict ({e})")
    policy_params = list(agent.parameters())
    backbone_params = list(model.parameters())
    
    # Print parameter counts
    def count_params(module, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())
    
    encoder_params = count_params(model.encoder)
    fusion_params = count_params(model.fusion)
    lstm_params = count_params(model.lstm)
    decoder_params = count_params(model.decoder)
    decoder_trainable = count_params(model.decoder, only_trainable=True)
    total_model_params = count_params(model)
    agent_params = count_params(agent)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Encoder:    {encoder_params:>12,} params ({encoder_params/1e6:.2f}M)")
    print(f"Fusion MLP: {fusion_params:>12,} params ({fusion_params/1e6:.2f}M)")
    print(f"LSTM:       {lstm_params:>12,} params ({lstm_params/1e6:.2f}M)")
    print(f"Decoder:    {decoder_params:>12,} params ({decoder_params/1e6:.2f}M) {'[FROZEN]' if decoder_trainable == 0 else ''}")
    print(f"{'-'*60}")
    print(f"Total Model:{total_model_params:>12,} params ({total_model_params/1e6:.2f}M)")
    print(f"Trainable:  {count_params(model, only_trainable=True):>12,} params ({count_params(model, only_trainable=True)/1e6:.2f}M)")
    print(f"Agent (RL): {agent_params:>12,} params ({agent_params/1e6:.2f}M)")
    print(f"{'-'*60}")
    print(f"GRAND TOTAL:{total_model_params + agent_params:>12,} params ({(total_model_params + agent_params)/1e6:.2f}M)")
    print("="*60 + "\n")

    # Lazily create policy optimizer when RL becomes enabled
    opt_policy = None
    global_rl_disabled = bool(no_rl)
    if global_rl_disabled:
        # Freeze policy/value heads when RL globally disabled
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
    # Gradient Difference Loss (GDL) for sharp edges
    gdl_weight = float(getattr(Config, 'GDL_WEIGHT', 0.0))
    criterion_gdl = GradientDifferenceLoss().to(Config.DEVICE) if gdl_weight > 0.0 else None
    # Foreground mask for L1 loss
    use_fg_mask = bool(getattr(Config, 'USE_FG_MASK', False))
    fg_thresh = float(getattr(Config, 'FG_THRESH', 0.1))
    bg_weight = float(getattr(Config, 'BG_WEIGHT', 0.1))

    # RL is always fully attached; no detach/ramp schedule

    # Schedule for freezing backbone during initial RL-only epochs, relative to RL start
    rl_only_epochs = int(getattr(Config, 'RL_ONLY_EPOCHS', 0)) if not global_rl_disabled else 0
    rl_start_epoch = 0 if global_rl_disabled else int(recon_warmup_epochs)
    backbone_frozen = False
    model.train()
    global_step = 0
    episodes_seen = 0
    for epoch in range(Config.EPOCHS):
        # Make sure we're in training mode at the start of each epoch
        model.train()
        agent.train()
        # Phase gating
        warmup_active = (epoch < recon_warmup_epochs)
        rl_enabled = (not global_rl_disabled) and (not warmup_active)
        cls_enabled = not (bool(no_cls_warmup) and warmup_active)

        # Manage RL optimizer lazily
        if rl_enabled and opt_policy is None:
            opt_policy = torch.optim.AdamW(
                policy_params,
                lr=getattr(Config, 'RL_POLICY_LR', Config.LEARNING_RATE),
                weight_decay=Config.WEIGHT_DECAY,
            )
            # Ensure agent params are trainable
            for p in agent.parameters():
                p.requires_grad = True

        # Backbone freeze schedule: freeze during [rl_start_epoch, rl_start_epoch + rl_only_epochs)
        if rl_only_epochs > 0 and rl_start_epoch <= epoch < rl_start_epoch + rl_only_epochs:
            if not backbone_frozen:
                print(f"[epoch {epoch}] RL-only phase: freezing backbone")
                for p in model.parameters():
                    p.requires_grad = False
                backbone_frozen = True
        else:
            if backbone_frozen:
                print(f"[epoch {epoch}] Unfreezing backbone after RL-only phase")
                for p in model.parameters():
                    p.requires_grad = True
                backbone_frozen = False

        # Unfreeze decoder after FREEZE_DECODER_EPOCHS from start of training
        if (hasattr(Config, 'FREEZE_DECODER_EPOCHS') and 
            Config.FREEZE_DECODER_EPOCHS > 0 and 
            epoch == Config.FREEZE_DECODER_EPOCHS):
            print(f"Unfreezing decoder at epoch {epoch}")
            for param in model.decoder.parameters():
                param.requires_grad = True
                
        for batch_idx, batch in enumerate(train_loader):
            # Support datasets that return (image, label, start_xy)
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                images, labels, starts_xy = batch
            else:
                images, labels = batch
                starts_xy = None
            total_rec_loss = torch.tensor(0.0, device=Config.DEVICE)
            image = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Start gaze position
            num_steps = Config.MAX_STEPS
            B = image.size(0)
            if starts_xy is not None:
                base = starts_xy.to(Config.DEVICE)
                jitter_r = float(getattr(Config, 'START_JITTER', 0.0) or 0.0)
                if jitter_r > 0.0:
                    jitter = (torch.rand(B, 2, device=Config.DEVICE) * 2.0 - 1.0) * jitter_r
                    base = base + jitter
                gaze = base.clamp(0.0, 1.0)
            elif hasattr(Config, 'START_GAZE') and getattr(Config, 'START_GAZE') is not None:
                base = torch.tensor(list(getattr(Config, 'START_GAZE')), device=Config.DEVICE).view(1, 2).repeat(B, 1)
                jitter_r = float(getattr(Config, 'START_JITTER', 0.0) or 0.0)
                if jitter_r > 0.0:
                    jitter = (torch.rand(B, 2, device=Config.DEVICE) * 2.0 - 1.0) * jitter_r
                    base = base + jitter
                gaze = base.clamp(0.0, 1.0)
            else:
                gaze = torch.rand(B, 2, device=Config.DEVICE) * 0.4 + 0.3  # central 40%

            # Initialize LSTM memory
            state = model.init_memory(B, Config.DEVICE)
            reconstruction = None

            attach_alpha = 1.0

            # No coupling schedule to log anymore

            # RL rollouts storage
            logprobs = []   # list of (B,)
            values = []     # list of (B,)
            entropies = []  # list of (B,)
            # record gaze positions per step (pre-action)
            gaze_path = []
            # Early stopping per-sample
            alive_mask = torch.ones(B, dtype=torch.bool, device=Config.DEVICE)
            last_step = torch.full((B,), -1, dtype=torch.long, device=Config.DEVICE)
            final_decision_logits = torch.zeros(B, 2, device=Config.DEVICE)

            # Prepare cumulative visibility mask for step-wise losses
            use_step_mask = bool(getattr(Config, "USE_MASKED_STEP_LOSS", False))
            if use_step_mask:
                H, W = Config.IMG_SIZE
                sum_mask = torch.zeros((B, 1, H, W), device=Config.DEVICE)
                sigma_base = Config.FOVEA_OUTPUT_SIZE[0] / max(H, W)
                sigma_step = sigma_base * float(getattr(Config, "STEP_MASK_SIGMA_SCALE", 0.35))

            # Accumulate per-step losses (optionally masked/local)
            for step in range(num_steps):
                # record current gaze before taking action
                # Keep full batch gaze history on device for final visibility mask
                gaze_path.append(gaze.detach().clone())
                # Multi-scale glimpse: k scales with sizes base*(2**i), all resized to FOVEA_OUTPUT_SIZE
                k_scales = int(getattr(Config, 'K_SCALES', 3))
                base_h, base_w = Config.FOVEA_CROP_SIZE
                patches = []
                for i in range(k_scales):
                    scale = 2 ** i
                    size = (base_h * scale, base_w * scale)
                    patches.append(crop_patch(image, gaze, size, resize_to=Config.FOVEA_OUTPUT_SIZE))
                reconstruction, state = model(patches, state, gaze)

                if use_step_mask:
                    # Build current-step mask and merge into cumulative visibility
                    step_mask = _make_gaussian_mask(H, W, gaze, sigma_frac=sigma_step, device=Config.DEVICE)  # (B,1,H,W)
                    sum_mask = torch.maximum(sum_mask, step_mask)
                    # Compute masked per-sample L1 over union mask
                    mask3 = sum_mask.expand(-1, 3, H, W)
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

                # Compute L1 loss alive subset (masked-only when enabled)
                alive_idx = torch.nonzero(alive_mask, as_tuple=False).squeeze(-1)
                if use_step_mask:
                    l1_step = l1_m
                else:
                    l1_step = l1_loss(reconstruction[alive_idx], image[alive_idx]) * weight

                # Compute MS-SSIM loss for this step if enabled
                if use_ssim:
                    mssim_step_val = ms_ssim(reconstruction, image, data_range=2.0, size_average=True)
                    ssim_step = (1.0 - mssim_step_val) * weight
                else:
                    ssim_step = 0.0

                # Optional GDL per-step on alive subset
                if criterion_gdl is not None:
                    gdl_step = criterion_gdl(reconstruction[alive_idx], image[alive_idx]) * weight
                else:
                    gdl_step = 0.0

                #only use masked loss
                gdl_step = 0.0

                rec_error = (
                    l1_weight * l1_step
                    + (ssim_weight * ssim_step if use_ssim else 0.0)
                    + (gdl_weight * gdl_step if criterion_gdl is not None else 0.0)
                )

                total_rec_loss = total_rec_loss + rec_error

                # Choose action: RL policy (move + stop) or uniform random over 8 directions
                if rl_enabled:
                    h_t = state[0][-1]  # (B,H)
                    move_logits, decision_logits_step, stop_logit, value_t = agent.full_policy(h_t, gaze)
                    cat = Categorical(logits=move_logits)
                    move_idx = cat.sample()               # (B,)
                    move_lp = cat.log_prob(move_idx)      # (B,)
                    move_ent = cat.entropy()              # (B,)
                    # Bernoulli stop
                    stop_prob = torch.sigmoid(stop_logit)
                    stop_dist = torch.distributions.Bernoulli(probs=stop_prob)
                    stop_sample = stop_dist.sample()       # (B,)
                    # Enforce a minimum number of moves before allowing stop
                    min_steps = int(getattr(Config, 'MIN_STEPS_BEFORE_STOP', 10))
                    if step < min(min_steps, num_steps) - 1:
                        stop_sample = torch.zeros_like(stop_sample)
                    # Confidence gating: only allow STOP if classifier confidence >= threshold
                    conf_thresh = float(getattr(Config, 'STOP_CONF_THRESH', 0.9))
                    if conf_thresh is not None and conf_thresh > 0.0:
                        with torch.no_grad():
                            conf = torch.softmax(decision_logits_step, dim=1).amax(dim=1)
                        stop_sample = torch.where(conf >= conf_thresh, stop_sample, torch.zeros_like(stop_sample))
                    stop_lp = stop_dist.log_prob(stop_sample)  # (B,)
                    # Compose joint logprob and entropy (approximate add)
                    logprob = move_lp + stop_lp
                    entropy = move_ent + (-(stop_prob * torch.log(stop_prob.clamp_min(1e-8)) + (1 - stop_prob) * torch.log((1 - stop_prob).clamp_min(1e-8))))
                    logprobs.append(logprob)
                    values.append(value_t)
                    entropies.append(entropy)
                    # final decision logits update for samples that stopped this step
                    newly_stopped = (stop_sample >= 0.5) & alive_mask
                    if newly_stopped.any():
                        final_decision_logits[newly_stopped] = decision_logits_step[newly_stopped]
                        last_step[newly_stopped] = step
                else:
                    # Dont stop when using random policy
                    move_idx = torch.randint(low=0, high=8, size=(gaze.shape[0],), device=Config.DEVICE)
                    stop_sample = torch.zeros(gaze.shape[0], device=Config.DEVICE)

                # Continuation mask for next step
                cont_mask = alive_mask & (stop_sample < 0.5)

                # Map discrete action index to delta and apply only for samples that continue
                deltas = eight_dir_deltas(Config.MAX_MOVE, device=Config.DEVICE)
                delta = deltas[move_idx]
                delta = delta * cont_mask.float().unsqueeze(1)

                # Apply action for next step
                gaze = torch.clamp(gaze + delta, 0.0, 1.0)
                # Keep gaze within central bounds if enabled
                if getattr(Config, 'USE_GAZE_BOUNDS', False):
                    frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
                    lo, hi = frac, 1.0 - frac
                    gaze = gaze.clamp(min=lo, max=hi)

                # Update alive mask and exit if all stopped
                alive_mask = cont_mask
                if alive_mask.sum() == 0:
                    break

            # Final reconstruction from final LSTM state with larger weight
            final_recon = model.decode_from_state(state)
            
            # Optional: restrict final reconstruction loss to what the agent actually observed
            use_final_mask = bool(getattr(Config, "USE_FINAL_VISIBILITY_MASK", False)) and len(gaze_path) > 0
            final_l1 = None
            masked_final_gdl = None
            if use_final_mask:
                H, W = Config.IMG_SIZE
                # Base sigma as fraction of image based on fovea size
                sigma_base = Config.FOVEA_OUTPUT_SIZE[0] ** float(getattr(Config, "K_SCALES", 3)) / max(H, W)
                sigma_frac_final = sigma_base * float(getattr(Config, "FINAL_MASK_SIGMA_SCALE", getattr(Config, "STEP_MASK_SIGMA_SCALE", 0.35)))
                # Build per-step masks and union across actually executed steps per sample
                T_obs = len(gaze_path)
                masks_t = []
                for t in range(T_obs):
                    masks_t.append(_make_gaussian_mask(H, W, gaze_path[t], sigma_frac=sigma_frac_final, device=Config.DEVICE))  # (B,1,H,W)
                masks_stack = torch.stack(masks_t, dim=0)  # (T,B,1,H,W)

                # Determine executed steps per-sample; include all steps if never stopped
                ls = last_step.clone()
                ls[ls < 0] = T_obs - 1
                t_idx = torch.arange(T_obs, device=Config.DEVICE).unsqueeze(1)  # (T,1)
                exec_masks_tb = (t_idx <= ls.unsqueeze(0)).to(masks_stack.dtype)  # (T,B)
                exec_masks_tb = exec_masks_tb.view(T_obs, -1, 1, 1, 1)
                masks_stack = masks_stack * exec_masks_tb
                # Union over time
                final_mask = torch.amax(masks_stack, dim=0)  # (B,1,H,W)
                # Expand to channels
                final_mask3 = final_mask.expand(-1, 3, H, W)

                # Masked per-sample L1 averaged over visible area only
                per_sample_l1 = (((final_recon - image).abs() * final_mask3).flatten(1).sum(dim=1) /
                                   (final_mask3.flatten(1).sum(dim=1) + 1e-6))
                final_l1 = per_sample_l1.mean()

                # If using GDL, use manual gdl inside mask
                if criterion_gdl is not None:
                    # Horizontal gradients: width-1
                    pred_dx = torch.abs(final_recon[:, :, :, 1:] - final_recon[:, :, :, :-1])
                    targ_dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
                    mask_dx = final_mask3[:, :, :, 1:] * final_mask3[:, :, :, :-1]
                    dx_diff = torch.abs(pred_dx - targ_dx) * mask_dx
                    dx_sum = mask_dx.flatten(1).sum(dim=1).clamp_min(1.0)  # avoid div by zero
                    dx_loss = (dx_diff.flatten(1).sum(dim=1) / dx_sum).mean()

                    # Vertical gradients: height-1
                    pred_dy = torch.abs(final_recon[:, :, 1:, :] - final_recon[:, :, :-1, :])
                    targ_dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
                    mask_dy = final_mask3[:, :, 1:, :] * final_mask3[:, :, :-1, :]
                    dy_diff = torch.abs(pred_dy - targ_dy) * mask_dy
                    dy_sum = mask_dy.flatten(1).sum(dim=1).clamp_min(1.0)
                    dy_loss = (dy_diff.flatten(1).sum(dim=1) / dy_sum).mean()
                    masked_final_gdl = dx_loss + dy_loss
            else:
                final_l1 = l1_loss(final_recon, image)
            
            # ANTI-COLLAPSE: Add variance penalty to final reconstruction
            final_var_penalty = 0.0
            final_variance = 0.0
            if getattr(Config, "USE_VARIANCE_PENALTY", False):
                recon_var = final_recon.var(dim=[2, 3]).mean()
                final_variance = recon_var.item()
                var_penalty = torch.exp(-recon_var * 10.0)
                final_var_penalty = var_penalty.item()
                final_l1 = final_l1 + getattr(Config, "VARIANCE_PENALTY_WEIGHT", 0.1) * var_penalty
            
            # ANTI-COLLAPSE: Penalize near-zero latent codes (prevents LSTM collapse)
            latent_norm_penalty = 0.0
            if getattr(Config, "USE_LATENT_NORM_PENALTY", False):
                # state is a tuple (hidden, cell) from LSTM, extract hidden state
                h_state = state[0] if isinstance(state, tuple) else state
                latent_norm = torch.norm(h_state, dim=1).mean()  # L2 norm across latent dimensions
                min_norm = getattr(Config, "LATENT_NORM_MIN", 0.5)
                norm_deficit = F.relu(min_norm - latent_norm)
                latent_norm_penalty = norm_deficit.item()
                penalty_weight = getattr(Config, "LATENT_NORM_PENALTY_WEIGHT", 1.0)
                final_l1 = final_l1 + norm_deficit * penalty_weight
            
            # Optional perceptual loss
            final_perc = None
            if criterion_perc is not None:
                final_perc = criterion_perc(final_recon, image)
            
            # Optional MS-SSIM loss
            final_ssim = None
            if use_ssim:
                # MS-SSIM expects inputs in [-1,1] range, set data_range=2.0
                mssim_val = ms_ssim(final_recon, image, data_range=2.0, size_average=True)
                final_ssim = 1.0 - mssim_val
            
            # Optional GDL on final reconstruction
            final_gdl = None
            if criterion_gdl is not None:
                if use_final_mask and masked_final_gdl is not None:
                    final_gdl = masked_final_gdl
                else:
                    final_gdl = criterion_gdl(final_recon, image)
            
            final_mult = getattr(Config, "FINAL_LOSS_MULT", 8.0)
            final_loss = l1_weight * final_l1 * final_mult
            if final_perc is not None and perc_weight > 0.0:
                final_loss = final_loss + perc_weight * final_perc * final_mult
            if final_ssim is not None and ssim_weight > 0.0:
                final_loss = final_loss + ssim_weight * final_ssim * final_mult
            if final_gdl is not None and gdl_weight > 0.0:
                final_loss = final_loss + gdl_weight * final_gdl * final_mult
            # Compute final decision logits for those never stopped
            T_exec = len(values)
            not_stopped = (last_step < 0)
            if not_stopped.any():
                h_T = state[0][-1]
                _mT, decision_logits_T, _sT, _vT = agent.full_policy(h_T, gaze)
                final_decision_logits[not_stopped] = decision_logits_T[not_stopped]
                last_step[not_stopped] = int(getattr(Config, 'MAX_STEPS', 20)) - 1
            # Classification loss on final decisions
            ce_cls = nn.CrossEntropyLoss()
            cls_loss = ce_cls(final_decision_logits, labels)

            total_loss = total_rec_loss + final_loss + (cls_loss if cls_enabled else 0.0)

            # ----- RL loss (A2C with GAE) -----
            rl_loss = torch.tensor(0.0, device=Config.DEVICE)
            if rl_enabled and len(values) > 0:
                gamma = float(getattr(Config, 'RL_GAMMA', 0.95))
                lam = float(getattr(Config, 'RL_LAMBDA', 0.95))
                scale = float(getattr(Config, 'RL_REWARD_SCALE', 1.0))
                preds = final_decision_logits.argmax(dim=1)
                r_final = (preds == labels).float() * scale  # (B,)

                # Stack per-step tensors with executed length T, then pad to MAX_STEPS for safe indexing/masking
                values_t_exec = torch.stack(values)      # (T,B)
                logprobs_t_exec = torch.stack(logprobs)  # (T,B)
                entropies_t_exec = torch.stack(entropies)  # (T,B)
                T_exec = values_t_exec.size(0)
                max_Steps = int(getattr(Config, 'MAX_STEPS', 20))

                def pad_to_max(t_TB):
                    if t_TB.size(0) == max_Steps:
                        return t_TB
                    pad_len = max_Steps - t_TB.size(0)
                    pad = torch.zeros((pad_len, *t_TB.shape[1:]), device=t_TB.device, dtype=t_TB.dtype)
                    return torch.cat([t_TB, pad], dim=0)

                values_t = pad_to_max(values_t_exec)          # (max_Steps,B)
                logprobs_t = pad_to_max(logprobs_t_exec)      # (max_Steps,B)
                entropies_t = pad_to_max(entropies_t_exec)    # (max_Steps,B)

                rewards_t = torch.zeros((max_Steps, labels.size(0)), device=Config.DEVICE)
                # Per-step time penalty to encourage shorter trajectories
                # Also additionally penalize taking less steps when final classification is wrong
                step_pen = float(getattr(Config, 'RL_STEP_PENALTY', 0.0))
                ls = last_step.clone()
                ls[ls < 0] = max_Steps - 1
                t_idx = torch.arange(max_Steps, device=Config.DEVICE).unsqueeze(1)  # (T,1)
                exec_masks_t = (t_idx <= ls.unsqueeze(0)).to(values_t.dtype)  # (max_Steps,B)
                cont_masks_t = (t_idx < ls.unsqueeze(0)).to(values_t.dtype)   # (max_Steps,B)
                # Per-step time penalty on each executed step
                if step_pen != 0.0:
                    rewards_t = rewards_t - step_pen * exec_masks_t
                # Add final reward at the last executed step for each sample
                batch_arange = torch.arange(labels.size(0), device=Config.DEVICE)
                ls_clamped = ls.clamp_min(0).clamp_max(max_Steps - 1)
                # Additional penalty for not executing all steps if final decision is wrong
                # This is scaled super high to strongly encourage running to full length when unsure
                incorrect = (preds != labels).float()
                stop_penalty = step_pen * incorrect * (max_Steps - 1 - ls_clamped).float()  * 200
                rewards_t[ls_clamped, batch_arange] = rewards_t[ls_clamped, batch_arange] + r_final - stop_penalty

                with torch.no_grad():
                    advantages = torch.zeros_like(values_t)
                    gae = torch.zeros_like(values_t[-1])
                    next_value = torch.zeros_like(values_t[-1])
                    for t in reversed(range(max_Steps)):
                        v_t = values_t[t]
                        v_tp1 = next_value
                        m_t = cont_masks_t[t]
                        delta = rewards_t[t] + gamma * v_tp1 * m_t - v_t
                        gae = delta + gamma * lam * m_t * gae
                        advantages[t] = gae
                        next_value = v_t
                    returns = advantages + values_t

                if bool(getattr(Config, 'RL_NORM_ADV', False)):
                    mask_sum = exec_masks_t.sum().clamp_min(1.0)
                    adv_mean = (advantages * exec_masks_t).sum() / mask_sum
                    adv_var = (((advantages - adv_mean) ** 2) * exec_masks_t).sum() / mask_sum
                    adv_std = adv_var.sqrt().clamp_min(1e-6)
                    advantages = (advantages - adv_mean) / adv_std

                denom = exec_masks_t.sum().clamp_min(1.0)
                policy_loss = -((logprobs_t * advantages.detach()) * exec_masks_t).sum() / denom
                value_loss = (((values_t - returns) ** 2) * exec_masks_t).sum() / denom
                entropy_loss = -(entropies_t * exec_masks_t).sum() / denom
                value_coef = float(getattr(Config, 'RL_VALUE_COEF', 0.5))
                entropy_coef = float(getattr(Config, 'RL_ENTROPY_COEF', 0.01))
                rl_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            total_loss = total_loss + float(getattr(Config, 'RL_LOSS_WEIGHT', 1.0)) * rl_loss

            # Backpropagation
            if opt_policy is not None and rl_enabled:
                opt_policy.zero_grad()
            # Allow stepping backbone except when explicitly frozen
            if not backbone_frozen:
                opt_backbone.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability (clip both backbone and agent)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
            if opt_policy is not None and rl_enabled:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=Config.GRAD_CLIP_NORM)

            # Step optimizers
            if opt_policy is not None and rl_enabled:
                opt_policy.step()
            if not backbone_frozen:
                opt_backbone.step()

            # Update counters and maybe checkpoint every 5000 images
            imgs_seen += images.size(0)
            if imgs_seen >= next_save_at:
                ckpt_path = os.path.join(run_dir, f"model_images_{imgs_seen}.pth")
                # Save agent only when RL is not globally disabled (i.e., not --no-rl)
                if not global_rl_disabled:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'agent_state_dict': agent.state_dict(),
                    }, ckpt_path)
                else:
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
                if 'final_gdl' in locals() and final_gdl is not None:
                    logs["loss/gdl_final"] = float(final_gdl.item())
                # Log classification loss
                if cls_enabled:
                    logs["loss/cls"] = float(cls_loss.item())
                logs["phase/warmup"] = int(warmup_active)
                logs["phase/rl_enabled"] = int(rl_enabled)
                wandb.log(logs, step=global_step)

            # Occasionally log a synchronized figure: Original+GazePath vs Reconstruction (every 10 batches)
            if batch_idx % 10 == 0:
                try:
                    H, W = Config.IMG_SIZE
                    # Prepare numpy images (convert from [-1,1] to [0,1] for display)
                    orig_np = ((image[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
                    # Use final_recon instead of intermediate reconstruction
                    recon_np = ((final_recon[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
                    # Compute final decision for first sample and confidence
                    with torch.no_grad():
                        probs = torch.softmax(final_decision_logits, dim=1)
                        pred0 = int(probs[0].argmax().item())
                        conf0 = float(probs[0, pred0].item())
                        true0 = int(labels[0].item())
                    # Build combined figure
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    # Left: Original with gaze path
                    axs[0].imshow(orig_np)
                    if len(gaze_path) > 0:
                        # Support both shapes: p is (2,) or (B,2); always visualize sample 0
                        pts = []
                        for p in gaze_path:
                            p0 = p if p.dim() == 1 else p[0]
                            pts.append((int(p0[0].item() * (W - 1)), int(p0[1].item() * (H - 1))))
                        xs, ys = zip(*pts)
                        axs[0].scatter(xs, ys, c='r', s=40, marker='x', label='gaze')
                        axs[0].plot(xs, ys, c='yellow', linewidth=1, alpha=0.8)
                    axs[0].set_title('Original + gaze path')
                    axs[0].axis('off')
                    # Right: Final step reconstruction
                    axs[1].imshow(recon_np)
                    axs[1].set_title('Reconstruction (final step)')
                    # Overlay prediction vs truth text box
                    txt = f"pred={pred0}  p={conf0*100:.1f}%  true={true0}"
                    axs[1].text(0.02, 0.98, txt, transform=axs[1].transAxes, va='top', ha='left',
                                color='white', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'), fontsize=9)
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
        # Save agent only when RL is not globally disabled (i.e., not --no-rl)
        if not global_rl_disabled:
            torch.save({
                'model_state_dict': model.state_dict(),
                'agent_state_dict': agent.state_dict(),
            }, model_path)
        else:
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
        
        # Skip expensive verification; handle bad files at load time
        print(f"Found {len(self.image_paths)} images (unverified) in {data_dir}")
    
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
    parser.add_argument('--recon-warmup-epochs', type=int, default=0,
                        help='Number of initial epochs with reconstruction-only (no RL); classification optionally disabled via --no-cls-warmup')
    parser.add_argument('--no-cls-warmup', action='store_true',
                        help='Disable classification loss during warmup epochs')
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
        train(
            use_pretrained_decoder=args.use_pretrained,
            load_full_model=args.load_full,
            no_rl=args.no_rl,
            wb=wb_run,
            recon_warmup_epochs=int(getattr(args, 'recon_warmup_epochs', 0) or 0),
            no_cls_warmup=bool(getattr(args, 'no_cls_warmup', False)),
        )
    finally:
        if wb_run is not None:
            wandb.finish()
