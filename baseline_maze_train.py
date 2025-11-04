import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
from typing import Tuple

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

from models import MazeCNNClassifier
from config import Config          
import json

def clear_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


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
        return img, torch.tensor(label, dtype=torch.float32)


def count_params(module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def train_classifier(wb=None):
    # Timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"baseline_maze_cnn_{timestamp}"
    run_dir = os.path.join("runs", log_name)
    os.makedirs(run_dir, exist_ok=True)

    # keep inputs in [0,1] as tensors;
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Dataset / loaders
    maze_root = getattr(Config, 'MAZE_ROOT', os.path.join('./Data', 'Maze15'))
    if not os.path.exists(os.path.join(maze_root, 'imgs', 'train')):
        raise FileNotFoundError(f"Maze dataset not found at {maze_root}. Set Config.MAZE_ROOT to your dataset root or generate it with Datasets/generate_maze.py")
    train_dataset = MazeDataset(maze_root, split='train', transform=transform_img)
    val_dataset = MazeDataset(maze_root, split='val', transform=transform_img)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / loss / optim
    model = MazeCNNClassifier().to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Print parameter count
    total_params = count_params(model)
    print("=" * 60)
    print("BASELINE CNN CLASSIFIER")
    print("=" * 60)
    print(f"Params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Device: {Config.DEVICE}")
    print("=" * 60)

    best_val_acc = 0.0
    global_step = 0
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels in train_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(Config, 'GRAD_CLIP_NORM', 1.0))
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                running_correct += int((preds == labels).sum().item())
                running_total += int(labels.numel())
                running_loss += float(loss.item()) * labels.size(0)

            if wb is not None:
                wandb.log({
                    "train/loss": float(loss.item()),
                }, step=global_step)
            global_step += 1

        train_loss = running_loss / max(1, running_total)
        train_acc = 100.0 * running_correct / max(1, running_total)
        val_loss, val_acc = evaluate(model, val_loader, Config.DEVICE)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")
        if wb is not None:
            wandb.log({
                "epoch": epoch,
                "metrics/train_loss": float(train_loss),
                "metrics/train_acc": float(train_acc),
                "metrics/val_loss": float(val_loss),
                "metrics/val_acc": float(val_acc),
            }, step=global_step)

        # Save best checkpoint by validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'IMG_SIZE': Config.IMG_SIZE,
                }
            }
            ckpt_path = os.path.join(run_dir, f"best_val_{best_val_acc:.2f}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved best checkpoint -> {ckpt_path}")

        # Periodic epoch checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch+1}.pth")
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

    # Final save
    final_path = os.path.join(run_dir, "final.pth")
    torch.save({'model_state_dict': model.state_dict()}, final_path)
    print(f"Training complete. Final model -> {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline CNN classifier on Maze dataset')
    parser.add_argument('--config', default='config_maze', help="config module name (e.g. 'config', 'config_maze')")
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='GazeControl', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    # Dynamically load config module and rebind Config used in this module
    if getattr(args, 'config', None):
        try:
            mod = importlib.import_module(args.config)
            Config = mod.Config  # rebind global used above
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Failed to load config module '{args.config}', falling back to default. Error: {e}")

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
                "model": "MazeCNNClassifier",
            }
        )

    try:
        train_classifier(wb=wb_run)
    finally:
        if wb_run is not None:
            wandb.finish()
        clear_memory()