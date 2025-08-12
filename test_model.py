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

from train import clear_memory    # reuse helper
from models import GazeControlModel
from foveal_blur import foveal_blur

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

def run_episode(model, image, device):
    image = image.to(device)
    # start at random gaze
    gaze = torch.rand(1,2,device=device)*0.5 + 0.25
    # initial foveation
    img_np = (image[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    cx,cy = int(gaze[0,0]*Config.IMG_SIZE[0]), int(gaze[0,1]*Config.IMG_SIZE[1])
    fovs, recs, centers = [], [], []              # add centers list
    # init LSTM state
    h = torch.zeros(1,Config.HIDDEN_SIZE,device=device)
    c = torch.zeros_like(h)
    for step in range(Config.MAX_STEPS):
        blurred = foveal_blur(img_np,(cx,cy),
                              output_size=Config.FOVEA_OUTPUT_SIZE,
                              crop_size=Config.FOVEA_CROP_SIZE)
        fov = torch.from_numpy(blurred.astype(np.float32)/255.0)\
                   .permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            rec, (delta, stop), _, (h,c) = model.sample_action(fov,h,c,gaze)
        fovs.append(fov.cpu()[0])
        recs.append(rec.cpu()[0])
        # update gaze
        delta = delta.clamp(-Config.MAX_MOVE, Config.MAX_MOVE)
        gaze = (gaze + delta).clamp(0,1)
        img_np = (image[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        cx,cy = int(gaze[0,0]*Config.IMG_SIZE[0]), int(gaze[0,1]*Config.IMG_SIZE[1])
        centers.append((cx, cy))                   # record center
        if stop.item()==1:
            pass
            #break
    return fovs, recs, centers                    # return centers

def visualize(original, fovs, recs, centers, out_png):
    steps = len(fovs)
    rows = steps+1; cols=2
    fig, axs = plt.subplots(rows,cols, figsize=(4*cols,4*rows))
    # row 0: original + blank
    axs[0,0].imshow(original.permute(1,2,0).cpu().numpy()); axs[0,0].set_title("Original"); axs[0,0].axis('off')
    axs[0,1].axis('off')
    # each subsequent row
    for i, (fov, rec) in enumerate(zip(fovs, recs), start=1):
        axs[i,0].imshow(fov.permute(1,2,0).numpy()); axs[i,0].set_title(f"Fovea {i}"); axs[i,0].axis('off')
        axs[i,1].imshow(rec.permute(1,2,0).numpy()); axs[i,1].set_title(f"Recon {i}"); axs[i,1].axis('off')
        cx, cy = centers[i-1]                      # lookup center
        axs[i,1].scatter(cx, cy, c='r', marker='x', s=100)  # mark fovea center
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
        hidden_size=Config.HIDDEN_SIZE,
        img_size=Config.IMG_SIZE,
        fovea_size=Config.FOVEA_OUTPUT_SIZE,
        memory_size=Config.MEMORY_SIZE,
        memory_dim=getattr(Config, 'MEMORY_DIM', 4)
    ).to(device)
    state = torch.load(chkpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = load_random_image(source)
    fovs, recs, centers = run_episode(model, img, device)             # unpack centers
    visualize(img[0], fovs, recs, centers, out_path)                  # pass centers

if __name__=="__main__":
    main()
