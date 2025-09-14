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
    gaze = torch.rand(1, 2, device=device) * 0.6 + 0.2

    # Init LSTM memory via model helper
    state = model.init_memory(1, device)

    fovs, recs, centers = [], [], []
    with torch.no_grad():
        for step in range(num_steps):
            # Crop and forward
            fov = crop_patch(image, gaze, Config.FOVEA_CROP_SIZE, resize_to=Config.FOVEA_OUTPUT_SIZE)
            rec, state = model(fov, state, gaze)

            fovs.append(fov.cpu()[0])
            recs.append(rec.cpu()[0])

            # Record pixel center for viz (x uses width, y uses height)
            H, W = Config.IMG_SIZE
            cx = int(gaze[0, 0].item() * (W - 1))
            cy = int(gaze[0, 1].item() * (H - 1))
            centers.append((cx, cy))

            # Policy step: get action from actor-critic and update gaze (skip after final if desired)
            h_t = state[0][-1]
            logits, _ = model.policy_value(h_t, gaze)
            if deterministic:
                action_idx = torch.argmax(logits, dim=-1)
            else:
                cat = Categorical(logits=logits)
                action_idx = cat.sample()
            deltas = eight_dir_deltas(Config.MAX_MOVE, device=Config.DEVICE)
            delta = deltas[action_idx]
            gaze = torch.clamp(gaze + delta, 0.0, 1.0)

        # Also compute final reconstruction from final state
        final_rec = model.decode_from_state(state).cpu()[0]

    return fovs, recs, centers, final_rec

def visualize(original, fovs, recs, centers, final_rec, out_png):
    steps = len(fovs)
    rows = steps + 1
    cols = 3  # Original, step recon, final recon (first row shows final)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))

    # Row 0: original, blank, final reconstruction
    axs[0, 0].imshow(original.permute(1, 2, 0).cpu().numpy()); axs[0, 0].set_title("Original"); axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(final_rec.permute(1, 2, 0).numpy()); axs[0, 2].set_title("Final Recon"); axs[0, 2].axis('off')

    # Each subsequent row: fovea patch and step reconstruction
    for i, (fov, rec) in enumerate(zip(fovs, recs), start=1):
        axs[i, 0].imshow(fov.permute(1, 2, 0).numpy()); axs[i, 0].set_title(f"Fovea {i}"); axs[i, 0].axis('off')
        axs[i, 1].imshow(rec.permute(1, 2, 0).numpy()); axs[i, 1].set_title(f"Recon {i}"); axs[i, 1].axis('off')
        # overlay gaze center on the step recon
        cx, cy = centers[i-1]
        axs[i, 1].scatter(cx, cy, c='r', marker='x', s=60)
        # third column left blank for alignment
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
    # Deterministic by default for repeatability; change False to sample
    fovs, recs, centers, final_rec = run_episode(model, img, device, deterministic=True)
    visualize(img[0], fovs, recs, centers, final_rec, out_path)

if __name__=="__main__":
    main()
