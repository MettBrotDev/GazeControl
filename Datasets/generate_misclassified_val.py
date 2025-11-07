"""
Generate only validation mazes that a given classifier misclassifies.

This mirrors Datasets/generate_maze.py but:
- Only produces a 'val' split
- Only saves samples for which the provided classifier prediction != ground-truth label

Classifier expectations:
- Model class is defined in models.py (e.g., MazeCNNClassifier or MazeCNNClassifierMedium)
- Checkpoint is a .pth/.pt from the baseline classifier training
- Input normalization: [0,1] with Resize(Config.IMG_SIZE) from the chosen config module

Usage
  python Datasets/generate_misclassified_val.py \
    --output-dir ./Data/Maze \
    --val-samples 1000 \
    --grid-size 6 --block-size 4 \
    --model-class MazeCNNClassifier \
    --checkpoint path/to/baseline.pth \
    --config_module config_maze
"""

import os
import sys
import argparse
import json
import time
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure repository root is on sys.path so we can import config_*.py and models.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib
import torch
from torchvision import transforms as T

# When run as `python Datasets/generate_misclassified_val.py`, the script's directory is on sys.path
# so we can import generate_maze helpers directly.
from generate_maze import MazeGenerator, save_dataset


def build_classifier(model_class: str, checkpoint: str, config_module: str):
    """Load classifier model and preprocessing.

    Returns: (model, device, transform)
    - model: torch.nn.Module in eval mode
    - device: the device used
    - transform: callable that maps a PIL image to a 1xC×H×W tensor in [0,1]
    """
    Config = importlib.import_module(config_module).Config
    device = Config.DEVICE
    import models as models_mod

    ModelCls = getattr(models_mod, model_class, None)
    if ModelCls is None:
        raise ValueError(f"Model class '{model_class}' not found in models.py")

    model = ModelCls().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    tfm = T.Compose([
        T.Resize(Config.IMG_SIZE),
        T.ToTensor(),  # [0,1]
    ])
    return model, device, tfm


def predict_label(model, device, tfm, np_image: np.ndarray) -> Tuple[int, float]:
    """Predict label using classifier.

    Args:
        model, device, tfm: from build_classifier
        np_image: (H,W,3) uint8

    Returns: (pred_label:int, prob_pos:float)
    """
    pil = Image.fromarray(np_image)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # (1,) or (1,1)
        logits = logits.view(1)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)
    return pred, float(prob)


def _start_end_touching(start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Return True if start and end grid cells touch (including diagonals).

    Touching is defined as Chebyshev distance 1: max(|dx|, |dy|) == 1.
    """
    sx, sy = start
    tx, ty = end
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    return (dx == 0 and dy == 1) or (dx == 1 and dy == 0) or (dx == 1 and dy == 1)


def main():
    ap = argparse.ArgumentParser(description='Generate val set of mazes misclassified by a given classifier')
    ap.add_argument('--output-dir', type=str, default='./Data/Maze', help='Dataset root')
    ap.add_argument('--val-samples', type=int, default=2000, help='Target number of misclassified validation samples to save')
    ap.add_argument('--grid-size', type=int, default=6, help='Grid size (blocks per dimension)')
    ap.add_argument('--block-size', type=int, default=4, help='Block size (pixels per block)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    ap.add_argument('--random-endpoints', action='store_true', help='Randomize start/end positions on edges for each maze')
    ap.add_argument('--balance', action='store_true', help='Attempt to alternate between connected/disconnected candidates')
    ap.add_argument('--max-attempts', type=int, default=None, help='Max candidate attempts; default auto = 50×val-samples')
    ap.add_argument('--model-class', type=str, default='MazeCNNClassifier', help='Classifier class name from models.py')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to classifier checkpoint (.pth/.pt)')
    ap.add_argument('--config_module', type=str, default='config_maze', help='Config module providing IMG_SIZE and DEVICE')
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Prepare generator and classifier
    gen = MazeGenerator(grid_size=args.grid_size, block_size=args.block_size, random_endpoints=bool(args.random_endpoints))
    model, device, tfm = build_classifier(args.model_class, args.checkpoint, args.config_module)

    target = int(args.val_samples)
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else target * 50

    images = []
    labels = []
    starts = []
    ends = []

    # Alternate attempt type if balance flag used
    want_connected = True

    pbar = tqdm(total=target, desc='Collected misclassified')
    attempts = 0
    while len(images) < target and attempts < max_attempts:
        attempts += 1
        # Generate candidate
        if args.balance:
            force_disconnected = not want_connected
            want_connected = not want_connected
        else:
            # 50/50 attempt
            force_disconnected = (np.random.rand() < 0.5)
        maze, connected, start, end = gen.generate_maze(force_disconnected=force_disconnected)
        # Skip trivial cases where start and end cells touch (including diagonals)
        if _start_end_touching(start, end):
            continue
        img = gen.maze_to_image(maze, start, end, mark_endpoints=True)
        gt = 1 if connected else 0

        # Classify
        pred, prob = predict_label(model, device, tfm, img)
        if pred != gt:
            images.append(img)
            labels.append(gt)
            starts.append(start)
            ends.append(end)
            pbar.update(1)
    pbar.close()

    if len(images) < target:
        print(f"Warning: only collected {len(images)} misclassified samples after {attempts} attempts (target was {target})")

    # Save to val split, mirroring generate_maze.py format
    save_dataset(images, labels, args.output_dir, split='val', starts=starts, ends=ends)
    print('Done.')


if __name__ == '__main__':
    main()
