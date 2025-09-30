import os
import argparse
import random
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import importlib          # changed
from torch.distributions import Categorical
import torch.nn.functional as F

from train import clear_memory, crop_patch, eight_dir_deltas    # reuse helpers consistent with training
from models import GazeControlModel

# Global Config placeholder
Config = None

def load_random_image(source, config=None):
    # Use passed config or fall back to global
    cfg = config or Config
    if cfg is None:
        raise ValueError("Config must be provided either as parameter or global variable")
    
    # define transforms
    t_img = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
    ])
    t_mnist = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1))
    ])
    if source=="mnist":
        ds = datasets.MNIST(root=cfg.MNIST_DATA_DIR, train=False,
                            transform=t_mnist, download=True)
    elif source=="cifar100":
        ds = datasets.CIFAR100(root=cfg.CIFAR100_DATA_DIR, train=False,
                               transform=t_img, download=True)
    else:
        try:
            ds = datasets.ImageFolder(root=cfg.LOCAL_DATA_DIR, transform=t_img)
        except FileNotFoundError:
            print(f"Warning: Local data directory '{cfg.LOCAL_DATA_DIR}' not found or has no class folders.")
            print("Falling back to CIFAR100 dataset...")
            ds = datasets.CIFAR100(root=cfg.CIFAR100_DATA_DIR, train=False,
                                   transform=t_img, download=True)
    
    idx = random.randrange(len(ds))
    img, _ = ds[idx]
    return img.unsqueeze(0)  # (1,3,H,W)

def run_episode(model, image, device, deterministic=False):
    """Run a rollout matching train.py's RL policy loop.
    Returns fovea patches, step reconstructions, centers, and final reconstruction.
    deterministic: if True, use policy mean action; else sample from Normal.
    """
    image = image.to(device)
    num_steps = Config.MAX_STEPS
    # Random initial gaze in central 60%
    if getattr(Config, 'USE_GAZE_BOUNDS', False):
        frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
        lo, hi = frac, 1.0 - frac
        gaze = torch.rand(1, 2, device=device) * (hi - lo) + lo
    else:
        gaze = torch.rand(1, 2, device=device) * 0.6 + 0.2

    # Init LSTM memory via model helper
    state = model.init_memory(1, device)

    fovs, recs, centers = [], [], []
    multi_patches = []  # list of lists of patches per step (largest->smallest)
    with torch.no_grad():
        for step in range(num_steps):
            # Multi-scale crop and forward (match training behavior)
            base_h, base_w = Config.FOVEA_CROP_SIZE
            k_scales = int(getattr(Config, 'K_SCALES', 3))
            patches = []
            for i in range(k_scales):
                scale = 2 ** i
                size = (base_h * scale, base_w * scale)
                patches.append(crop_patch(image, gaze, size, resize_to=Config.FOVEA_OUTPUT_SIZE))
            # Keep copies for visualization ordered largest->smallest for composite
            fov = patches[0]
            multi_patches.append([p.cpu()[0] for p in reversed(patches)])
            rec, state = model(patches, state, gaze)

            fovs.append(fov.cpu()[0])
            recs.append(rec.cpu()[0])

            # Record pixel center for viz (x uses width, y uses height)
            H, W = Config.IMG_SIZE
            cx = int(gaze[0, 0].item() * (W - 1))
            cy = int(gaze[0, 1].item() * (H - 1))
            centers.append((cx, cy))

            # Step the gaze: for testing, use random action from the discrete set (no trained agent assumed)
            if deterministic:
                # deterministic: pick a fixed direction (e.g., East) for reproducibility
                action_idx = torch.tensor([0], device=device)  # 0 -> E
            else:
                action_idx = torch.randint(low=0, high=8, size=(1,), device=device)
            deltas = eight_dir_deltas(Config.MAX_MOVE, device=Config.DEVICE)
            delta = deltas[action_idx]
            gaze = torch.clamp(gaze + delta, 0.0, 1.0)
            if getattr(Config, 'USE_GAZE_BOUNDS', False):
                frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
                lo, hi = frac, 1.0 - frac
                gaze = gaze.clamp(min=lo, max=hi)

        # Also compute final reconstruction from final state
        final_rec = model.decode_from_state(state).cpu()[0]

    return fovs, recs, centers, final_rec, multi_patches

def _composite_centered(patches, base_upscale: int = 4):
    """Centered overlay from [p_large, p_mid, p_small], each (C,9,9), displayed at different sizes.
    - Each patch keeps 9x9 pixels but is upscaled with nearest-neighbor to different display sizes.
    - Canvas size = 9 * base_upscale (default 36). Overlay sizes = [1.0, 0.5, 0.25] of canvas.
    Returns numpy HxWx3 image in [0,1]."""
    if len(patches) == 0:
        return None
    C, H, W = patches[0].shape
    canvas_size = int(9 * base_upscale)
    canvas = torch.zeros(3, canvas_size, canvas_size)
    scales = [1.0, 0.5, 0.25]
    colors = [
        torch.tensor([1.0, 1.0, 1.0]),  # large - white
        torch.tensor([1.0, 1.0, 0.0]),  # mid - yellow
        torch.tensor([1.0, 0.0, 0.0]),  # small - red
    ]
    outline_stride = 2  # dashed effect to appear thinner
    outline_alpha = 0.6
    for i, p in enumerate(patches[:3]):
        target = max(1, int(round(canvas_size * scales[i])))
        # Always upsample from 9x9 to target using nearest to preserve pixel grid
        p_res = F.interpolate(p.unsqueeze(0), size=(target, target), mode='nearest')[0]
        y0 = (canvas_size - target) // 2
        x0 = (canvas_size - target) // 2
        canvas[:, y0:y0+target, x0:x0+target] = p_res
        # Draw dashed 1px outline with alpha blend for thinner look
        col = colors[i % len(colors)]
        col3x1 = col.view(3, 1)
        # top
        sl = canvas[:, y0, x0:x0+target:outline_stride]
        canvas[:, y0, x0:x0+target:outline_stride] = (1 - outline_alpha) * sl + outline_alpha * col3x1
        # bottom
        sl = canvas[:, y0+target-1, x0:x0+target:outline_stride]
        canvas[:, y0+target-1, x0:x0+target:outline_stride] = (1 - outline_alpha) * sl + outline_alpha * col3x1
        # left
        sl = canvas[:, y0:y0+target:outline_stride, x0]
        canvas[:, y0:y0+target:outline_stride, x0] = (1 - outline_alpha) * sl + outline_alpha * col3x1
        # right
        sl = canvas[:, y0:y0+target:outline_stride, x0+target-1]
        canvas[:, y0:y0+target:outline_stride, x0+target-1] = (1 - outline_alpha) * sl + outline_alpha * col3x1
    return canvas.permute(1,2,0).numpy()

def visualize(original, fovs, recs, centers, final_rec, out_png, multi_patches=None):
    steps = len(fovs)
    rows = steps + 1
    cols = 3  # Original, composite/step recon, final recon with path
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))

    # Row 0: original and final reconstruction with path
    axs[0, 0].imshow(original.permute(1, 2, 0).cpu().numpy()); axs[0, 0].set_title("Original"); axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(final_rec.permute(1, 2, 0).numpy()); axs[0, 2].set_title("Final Recon (path)"); axs[0, 2].axis('off')
    # overlay full path on final reconstruction
    if centers:
        xs = [x for (x, y) in centers]
        ys = [y for (x, y) in centers]
        axs[0, 2].plot(xs, ys, color='cyan', linewidth=2, alpha=0.8)
        axs[0, 2].scatter(xs, ys, c='cyan', s=10, alpha=0.8)
        axs[0, 2].scatter(xs[0], ys[0], c='lime', s=40, marker='o', label='start')
        axs[0, 2].scatter(xs[-1], ys[-1], c='red', s=40, marker='x', label='end')
        axs[0, 2].legend(loc='lower right', fontsize=8)

    # Each subsequent row: composite and step reconstruction (no fovea)
    for i, (fov, rec) in enumerate(zip(fovs, recs), start=1):
        # Composite of three scales (if available)
        comp = None
        if multi_patches is not None and len(multi_patches) >= i:
            comp = _composite_centered(multi_patches[i-1], base_upscale=4)
        if comp is not None:
            axs[i, 0].imshow(comp, interpolation='nearest'); axs[i, 0].set_title(f"Composite {i}")
        else:
            axs[i, 0].axis('off')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(rec.permute(1, 2, 0).numpy()); axs[i, 1].set_title(f"Recon {i}"); axs[i, 1].axis('off')
        # overlay current gaze center on the step recon
        cx, cy = centers[i-1]
        axs[i, 1].scatter(cx, cy, c='r', marker='x', s=60)
        # third column left blank for alignment on step rows
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved visualization to {out_png}")

def main():
    global Config  # ensure we can modify the global Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", default="config",
                        help="config module name (e.g. 'config' or 'configSmol')")
    parser.add_argument("--checkpoint", required=True, help="path to .pth")
    parser.add_argument("--source", choices=["mnist","local","cifar100"], default=None,
                        help="override DATA_SOURCE from config")
    parser.add_argument("--out", default="test_episode.png")
    args = parser.parse_args()

    # handle PastRuns folder for checkpoints and outputs
    past_runs_dir = os.path.join(os.path.dirname(__file__), "PastRuns")
    os.makedirs(past_runs_dir, exist_ok=True)
    chkpt = args.checkpoint
    if not os.path.isabs(chkpt):
        chkpt = os.path.join(past_runs_dir, chkpt)
    model_name = os.path.splitext(os.path.basename(chkpt))[0]
    out_name = f"{model_name}_{args.out}"
    out_path = os.path.join(past_runs_dir, out_name)

    # import chosen config module by name and set global Config
    config_module = importlib.import_module(args.config_module)
    Config = config_module.Config
    
    # Debug print to verify config loading
    print(f"Loaded config from {args.config_module}: IMG_SIZE={Config.IMG_SIZE}, FOVEA_OUTPUT_SIZE={Config.FOVEA_OUTPUT_SIZE}")

    # use CLI override or fall back to config
    source = args.source or Config.DATA_SOURCE

    device = Config.DEVICE
    model = GazeControlModel(
        encoder_output_size=Config.ENCODER_OUTPUT_SIZE,
        state_size=Config.HIDDEN_SIZE,
        img_size=Config.IMG_SIZE,
        fovea_size=Config.FOVEA_OUTPUT_SIZE,
        pos_encoding_dim=Config.POS_ENCODING_DIM,
        lstm_layers=Config.LSTM_LAYERS,
        decoder_latent_ch=Config.DECODER_LATENT_CH,
        k_scales=getattr(Config, 'K_SCALES', 3),
        fuse_to_dim=getattr(Config, 'FUSION_TO_DIM', None),
        fusion_hidden_mul=getattr(Config, 'FUSION_HIDDEN_MUL', 2.0),
    ).to(device)
    raw = torch.load(chkpt, map_location=device)
    # Accept common checkpoint formats
    if isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
    elif isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    else:
        state = raw
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"Strict load failed ({e}); retrying with strict=False")
        res = model.load_state_dict(state, strict=False)
        print(f"Loaded with missing={len(getattr(res,'missing_keys',[]))} unexpected={len(getattr(res,'unexpected_keys',[]))}")
    model.eval()

    img = load_random_image(source, config=Config)
    # Use random actions by default for more natural movement; set deterministic=True to fix direction
    fovs, recs, centers, final_rec, multi_patches = run_episode(model, img, device, deterministic=False)
    visualize(img[0], fovs, recs, centers, final_rec, out_path, multi_patches=multi_patches)

if __name__=="__main__":
    main()
