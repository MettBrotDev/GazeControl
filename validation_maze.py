import os
# silence TF/XLA plugin re‐registration messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import datetime
import argparse
import importlib
import json
import glob

from models import GazeControlModel, Agent
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
            meta = json.load(f)
        self.samples = [(m['filename'], int(m['label'])) for m in meta]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        p = os.path.join(self.img_dir, fname)
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def validate(
    split: str = 'val',
    model_path: str | None = None,
):
    # Setup transforms (match training range [-1,1])
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    # Dataset/loader
    maze_root = getattr(Config, 'MAZE_ROOT', os.path.join('./Data', 'Maze15'))
    if not os.path.exists(os.path.join(maze_root, 'imgs', split)):
        raise FileNotFoundError(f"Maze split '{split}' not found at {maze_root}. Set Config.MAZE_ROOT or generate it with Datasets/generate_maze.py")
    dataset = MazeDataset(maze_root, split=split, transform=transform_img)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model and agent
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
    agent = Agent(state_size=Config.HIDDEN_SIZE,
                  pos_encoding_dim=Config.POS_ENCODING_DIM,
                  stop_init_bias=float(getattr(Config, 'RL_STOP_INIT_BIAS', -8.0))).to(Config.DEVICE)

    # Load checkpoint (supports multiple formats)
    ckpt_path = model_path or getattr(Config, 'PRETRAINED_MODEL_PATH', '')
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found. Provide --model-path or set Config.PRETRAINED_MODEL_PATH. Tried: '{ckpt_path}'")
    raw = torch.load(ckpt_path, map_location=Config.DEVICE)
    try:
        if isinstance(raw, dict) and ('model_state_dict' in raw or 'agent_state_dict' in raw):
            if 'model_state_dict' in raw:
                model.load_state_dict(raw['model_state_dict'], strict=False)
            if 'agent_state_dict' in raw:
                agent.load_state_dict(raw['agent_state_dict'], strict=False)
        elif isinstance(raw, dict) and 'state_dict' in raw:
            model.load_state_dict(raw['state_dict'], strict=False)
        else:
            # Assume raw is a plain model state_dict
            model.load_state_dict(raw, strict=False)
    except Exception as e:
        print(f"Warning: failed to load checkpoint strictly ({e}); attempting non-strict load if possible.")
        try:
            if isinstance(raw, dict) and 'model_state_dict' in raw:
                model.load_state_dict(raw['model_state_dict'], strict=False)
            elif isinstance(raw, dict) and 'state_dict' in raw:
                model.load_state_dict(raw['state_dict'], strict=False)
            else:
                model.load_state_dict(raw, strict=False)
        except Exception as e2:
            print(f"Error: Could not load model weights: {e2}")
    model.eval(); agent.eval()

    # Only compute classification accuracy (mirror train_maze rollout, but greedy actions)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            image = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            B = image.size(0)
            # Init gaze
            if hasattr(Config, 'START_GAZE') and getattr(Config, 'START_GAZE') is not None:
                base = torch.tensor(list(getattr(Config, 'START_GAZE')), device=Config.DEVICE).view(1, 2).repeat(B, 1)
                jitter_r = float(getattr(Config, 'START_JITTER', 0.0) or 0.0)
                if jitter_r > 0.0:
                    jitter = (torch.rand(B, 2, device=Config.DEVICE) * 2.0 - 1.0) * jitter_r
                    base = base + jitter
                gaze = base.clamp(0.0, 1.0)
            else:
                gaze = torch.rand(B, 2, device=Config.DEVICE) * 0.4 + 0.3

            # RNN state
            state = model.init_memory(B, Config.DEVICE)
            num_steps = int(getattr(Config, 'MAX_STEPS', 20))

            # Track per-sample stopping and decisions (mirror train_maze)
            alive_mask = torch.ones(B, dtype=torch.bool, device=Config.DEVICE)
            last_step = torch.full((B,), -1, dtype=torch.long, device=Config.DEVICE)
            final_decision_logits = torch.zeros(B, 2, device=Config.DEVICE)

            for step in range(num_steps):
                # Multi-scale glimpse
                k_scales = int(getattr(Config, 'K_SCALES', 3))
                base_h, base_w = Config.FOVEA_CROP_SIZE
                patches = []
                for i in range(k_scales):
                    scale = 2 ** i
                    size = (base_h * scale, base_w * scale)
                    patches.append(crop_patch(image, gaze, size, resize_to=Config.FOVEA_OUTPUT_SIZE))
                _recon, state = model(patches, state, gaze)

                # If none alive, break
                alive_idx = torch.nonzero(alive_mask, as_tuple=False).squeeze(-1)
                if alive_idx.numel() == 0:
                    break

                # Greedy policy: choose argmax move and stop if prob>=0.5
                h_t = state[0][-1]
                move_logits, decision_logits_step, stop_logit, _value_t = agent.full_policy(h_t, gaze)
                action_idx = move_logits.argmax(dim=1)
                stop_prob = torch.sigmoid(stop_logit)
                stop_sample = (stop_prob >= 0.5).float()
                # Enforce a minimum number of moves before allowing stop
                min_steps = int(getattr(Config, 'MIN_STEPS_BEFORE_STOP', 10))
                if step < min(min_steps, int(getattr(Config, 'MAX_STEPS', 20))) - 1:
                    stop_sample = torch.zeros_like(stop_sample)

                # Record decisions for samples that stop now
                newly_stopped = (stop_sample >= 0.5) & alive_mask
                if newly_stopped.any():
                    final_decision_logits[newly_stopped] = decision_logits_step[newly_stopped]
                    last_step[newly_stopped] = step

                # Continue mask
                cont_mask = alive_mask & (stop_sample < 0.5)

                # Apply action only for continuing samples
                deltas = eight_dir_deltas(Config.MAX_MOVE, device=Config.DEVICE)
                delta = deltas[action_idx]
                delta = delta * cont_mask.float().unsqueeze(1)
                gaze = torch.clamp(gaze + delta, 0.0, 1.0)
                if getattr(Config, 'USE_GAZE_BOUNDS', False):
                    frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
                    lo, hi = frac, 1.0 - frac
                    gaze = gaze.clamp(min=lo, max=hi)

                # Update alive mask; exit if all stopped
                alive_mask = cont_mask
                if alive_mask.sum() == 0:
                    break

            # For those that never stopped, take final decision at T
            not_stopped = (last_step < 0)
            if not_stopped.any():
                h_T = state[0][-1]
                _mT, decision_logits_T, _sT, _vT = agent.full_policy(h_T, gaze)
                final_decision_logits[not_stopped] = decision_logits_T[not_stopped]
                last_step[not_stopped] = num_steps - 1

            preds = final_decision_logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    acc = 100.0 * correct / max(1, total)
    print(f"Validation complete: split={split}, accuracy={acc:.2f}% ({correct}/{total})")

    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate GazeControl model on Maze dataset (accuracy only)')
    parser.add_argument('--config', default='config', help="config module name (e.g. 'config', 'config_maze')")
    parser.add_argument('--model-path', type=str, default=None, help='Path to checkpoint (supports model+agent dict)')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to validate on')
    args = parser.parse_args()

    # Dynamically load config module and rebind Config used in this module
    if getattr(args, 'config', None) and args.config != 'config':
        try:
            mod = importlib.import_module(args.config)
            Config = mod.Config  # rebind global used above
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Failed to load config module '{args.config}', falling back to default. Error: {e}")

    try:
        validate(
            split=str(getattr(args, 'split', 'val') or 'val'),
            model_path=args.model_path,
        )
    finally:
        clear_memory()
