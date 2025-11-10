import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models import MazeCNNClassifier
from config import Config
import importlib
from typing import Tuple

class MazeDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'val', transform=None):
        self.root = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'imgs', split)
        self.meta_path = os.path.join(root_dir, f'{split}_metadata.json')
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        self.samples = [(m['filename'], int(m['label'])) for m in meta]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
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

def main():
    parser = argparse.ArgumentParser(description='Validate baseline MazeCNNClassifier')
    parser.add_argument('--config', default='config_maze', help="Config module name")
    parser.add_argument('--data-root', type=str, default=None, help='Override Config.MAZE_ROOT')
    parser.add_argument('--split', type=str, default='val', choices=['train','val','test'], help='Dataset split to evaluate')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override Config.BATCH_SIZE for evaluation')
    args = parser.parse_args()

    # Load config module dynamically if provided
    if getattr(args, 'config', None):
        try:
            mod = importlib.import_module(args.config)
            ConfigClass = mod.Config
        except Exception as e:
            print(f"Failed to load config '{args.config}': {e}. Falling back to default Config.")
            ConfigClass = Config
    else:
        ConfigClass = Config

    device = getattr(ConfigClass, 'DEVICE', 'cpu')
    maze_root = args.data_root or getattr(ConfigClass, 'MAZE_ROOT', './Data/Maze')
    if not os.path.exists(os.path.join(maze_root, 'imgs', args.split)):
        raise FileNotFoundError(f"Maze dataset split '{args.split}' not found under {maze_root}.")

    batch_size = args.batch_size or getattr(ConfigClass, 'BATCH_SIZE', 64)

    transform_img = transforms.Compose([
        transforms.Resize(getattr(ConfigClass, 'IMG_SIZE', 24)),
        transforms.ToTensor(),
    ])

    dataset = MazeDataset(maze_root, split=args.split, transform=transform_img)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load model
    model = MazeCNNClassifier().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")

    val_loss, val_acc = evaluate(model, loader, device)
    print(f"Split: {args.split}  loss={val_loss:.4f}  acc={val_acc:.2f}%  samples={len(dataset)}")

if __name__ == '__main__':
    main()
