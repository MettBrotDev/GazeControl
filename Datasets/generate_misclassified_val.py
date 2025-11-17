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
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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


# Top-level worker to generate a single candidate (picklable for multiprocessing)
def _make_candidate(
    grid_size: int,
    block_size: int,
    random_endpoints: bool,
    force_disconnected: bool,
    seed: Optional[int],
    task_idx: int,
):
    if seed is not None:
        # Simple deterministic perturbation per task
        np.random.seed(int((seed + 9973 * task_idx) % (2**32 - 1)))
    gen = MazeGenerator(grid_size=grid_size, block_size=block_size, random_endpoints=bool(random_endpoints))
    maze, connected, start, end, plen = gen.generate_maze(force_disconnected=force_disconnected)
    img = gen.maze_to_image(maze, start, end, mark_endpoints=True)
    gt = 1 if connected else 0
    return img, gt, start, end, (plen if connected else -1)


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
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size for classifier inference')
    ap.add_argument('--workers', type=int, default=0, help='Number of parallel workers for candidate generation (0 = no multiprocessing)')
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

    images: List[np.ndarray] = []
    labels: List[int] = []
    starts: List[Tuple[int,int]] = []
    ends: List[Tuple[int,int]] = []
    path_lengths: List[int] = []

    # Alternate attempt type if balance flag used
    want_connected = True

    pbar = tqdm(total=target, desc='Collected misclassified')
    attempts = 0

    def classify_batch(batch_imgs: List[np.ndarray], batch_meta: List[Tuple[int, Tuple[int,int], Tuple[int,int], int]]):
        if not batch_imgs:
            return 0
        xs = []
        for img in batch_imgs:
            pil = Image.fromarray(img)
            xs.append(tfm(pil))
        x = torch.stack(xs, dim=0).to(device)
        with torch.no_grad():
            logits = model(x).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        collected = 0
        for i, (gt, start, end, plen) in enumerate(batch_meta):
            pred = int(probs[i] >= 0.5)
            if pred != gt:
                images.append(batch_imgs[i])
                labels.append(gt)
                starts.append(start)
                ends.append(end)
                path_lengths.append(plen)
                collected += 1
                pbar.update(1)
                if len(images) >= target:
                    break
        return collected

    if args.workers and args.workers > 0:
        # Parallel candidate generation, batched classification
        max_workers = int(args.workers)
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
            outstanding = {}
            submitted = 0
            batch_imgs: List[np.ndarray] = []
            batch_meta: List[Tuple[int, Tuple[int,int], Tuple[int,int], int]] = []

            def maybe_submit():
                nonlocal submitted, want_connected
                if args.balance:
                    force_disconnected = not want_connected
                    want_connected = not want_connected
                else:
                    force_disconnected = (np.random.rand() < 0.5)
                fut = ex.submit(
                    _make_candidate,
                    args.grid_size,
                    args.block_size,
                    bool(args.random_endpoints),
                    force_disconnected,
                    args.seed,
                    submitted,
                )
                outstanding[fut] = True
                submitted += 1

            # Prime pipeline
            while len(outstanding) < max_workers * 2 and attempts < max_attempts:
                maybe_submit()
                attempts += 1

            while len(images) < target and attempts <= max_attempts:
                if not outstanding:
                    break
                for fut in as_completed(list(outstanding.keys())):
                    outstanding.pop(fut, None)
                    try:
                        img, gt, start, end, plen = fut.result()
                    except Exception:
                        # Skip failed task
                        img = None
                    if img is not None and not _start_end_touching(start, end):
                        batch_imgs.append(img)
                        batch_meta.append((gt, start, end, plen))
                    # Submit more to keep pipeline filled
                    if attempts < max_attempts:
                        maybe_submit()
                        attempts += 1
                    # Classify if batch is ready or nearly done
                    if len(batch_imgs) >= args.batch_size or (len(images) + len(batch_imgs)) >= target:
                        classify_batch(batch_imgs, batch_meta)
                        batch_imgs.clear(); batch_meta.clear()
                    if len(images) >= target:
                        break
                # End for fut
            # Flush remaining
            if len(images) < target and batch_imgs:
                classify_batch(batch_imgs, batch_meta)
                batch_imgs.clear(); batch_meta.clear()
    else:
        # Sequential generation, batched classification
        batch_imgs: List[np.ndarray] = []
        batch_meta: List[Tuple[int, Tuple[int,int], Tuple[int,int], int]] = []
        while len(images) < target and attempts < max_attempts:
            attempts += 1
            if args.balance:
                force_disconnected = not want_connected
                want_connected = not want_connected
            else:
                force_disconnected = (np.random.rand() < 0.5)
            maze, connected, start, end, plen = gen.generate_maze(force_disconnected=force_disconnected)
            if _start_end_touching(start, end):
                continue
            img = gen.maze_to_image(maze, start, end, mark_endpoints=True)
            gt = 1 if connected else 0
            batch_imgs.append(img)
            batch_meta.append((gt, start, end, (plen if connected else -1)))
            if len(batch_imgs) >= args.batch_size:
                classify_batch(batch_imgs, batch_meta)
                batch_imgs.clear(); batch_meta.clear()
                if len(images) >= target:
                    break
        # Flush remaining
        if len(images) < target and batch_imgs:
            classify_batch(batch_imgs, batch_meta)
            batch_imgs.clear(); batch_meta.clear()

    if len(images) < target:
        print(f"Warning: only collected {len(images)} misclassified samples after {attempts} attempts (target was {target})")

    # Save to val split, mirroring generate_maze.py format
    save_dataset(images, labels, args.output_dir, split='val', starts=starts, ends=ends, path_lengths=path_lengths)
    print('Done.')


if __name__ == '__main__':
    main()
