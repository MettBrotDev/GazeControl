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

def load_random_image(source):
    # define transforms
    t_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
    ])
    t_mnist = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1))
    ])
    if source=="mnist":
        ds = datasets.MNIST(root=Config.MNIST_DATA_DIR, train=False,
                            transform=t_mnist, download=True)
    else:
        ds = datasets.ImageFolder(root=Config.LOCAL_DATA_DIR, transform=t_img)
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
    fovs, recs = [], []
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
        if stop.item()==1:
            pass
            #break
    return fovs, recs

def visualize(original, fovs, recs, out_png):
    steps = len(fovs)
    rows = steps+1; cols=2
    fig, axs = plt.subplots(rows,cols, figsize=(4*cols,4*rows))
    # row 0: original + blank
    axs[0,0].imshow(original.permute(1,2,0).cpu().numpy()); axs[0,0].set_title("Original"); axs[0,0].axis('off')
    axs[0,1].axis('off')
    # each subsequent row
    for i,(fov,rec) in enumerate(zip(fovs,recs),start=1):
        axs[i,0].imshow(fov.permute(1,2,0).numpy()); axs[i,0].set_title(f"Fovea {i}"); axs[i,0].axis('off')
        axs[i,1].imshow(rec.permute(1,2,0).numpy()); axs[i,1].set_title(f"Recon {i}"); axs[i,1].axis('off')
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved visualization to {out_png}")

def main():
    global Config  # ensure we can modify the global Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", default="config",
                        help="config module name (e.g. 'config' or 'configSmol')")
    parser.add_argument("--checkpoint", required=True, help="path to .pth")
    parser.add_argument("--source", choices=["mnist","local"], default=None,
                        help="override DATA_SOURCE from config")
    parser.add_argument("--out", default="test_episode.png")
    args = parser.parse_args()

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
        fovea_size=Config.FOVEA_OUTPUT_SIZE
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = load_random_image(source)
    fovs, recs = run_episode(model, img, device)
    visualize(img[0], fovs, recs, args.out)

if __name__=="__main__":
    main()
