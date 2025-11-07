import argparse
import importlib
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import ttk


def to01(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] clamp."""
    return ((t + 1.0) * 0.5).clamp(0.0, 1.0)


def crop_patch(image: torch.Tensor, gaze: torch.Tensor, crop_size: Tuple[int, int], resize_to: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Crop a patch around the gaze from a batched image tensor.
    image: (B,3,H,W) in [-1,1]
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
            # zero pad out-of-bounds regions
            crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

        # Ensure exact target crop size (in case of rounding)
        if crop.shape[-2:] != (ch, cw):
            crop = F.interpolate(crop, size=(ch, cw), mode='bilinear', align_corners=False)

        crops.append(crop)

    crops = torch.cat(crops, dim=0)
    if resize_to is not None and (resize_to[0] != ch or resize_to[1] != cw):
        crops = F.interpolate(crops, size=resize_to, mode='bilinear', align_corners=False)
    return crops


def build_glimpses(img_path: str, Config, xy: Tuple[float, float], vis_scale: Optional[int] = None):
    # Transforms: resize to model size, to tensor, map to [-1,1]
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)  # (1,3,H,W)

    # Normalize provided coords into [0,1]
    gx, gy = float(xy[0]), float(xy[1])
    if gx > 1.0 or gy > 1.0:
        # Treat as pixel coords w.r.t. Config.IMG_SIZE
        W = Config.IMG_SIZE[1]
        H = Config.IMG_SIZE[0]
        gx = np.clip(gx / (W - 1), 0.0, 1.0)
        gy = np.clip(gy / (H - 1), 0.0, 1.0)
    gaze = torch.tensor([[gx, gy]], dtype=torch.float32)

    # Multi-scale glimpses
    k_scales = int(getattr(Config, 'K_SCALES', 3))
    base_h, base_w = Config.FOVEA_CROP_SIZE
    out_h, out_w = Config.FOVEA_OUTPUT_SIZE
    patches = []
    for i in range(k_scales):
        scale = 2 ** i
        size = (base_h * scale, base_w * scale)
        p = crop_patch(img_t, gaze, size, resize_to=(out_h, out_w))  # (1,3,out_h,out_w)
        patches.append(p[0])  # (3,h,w)

    # Convert to [0,1] and upscale for visibility
    if vis_scale is None:
        vis_scale = int(getattr(Config, 'VIEWER_GLI_SCALE', 64))  # e.g., 4x4 -> 256x256
    vis = []
    for p in patches:
        p01 = to01(p.unsqueeze(0))  # (1,3,h,w)
        p_big = F.interpolate(p01, size=(out_h * vis_scale, out_w * vis_scale), mode='nearest')
        vis.append(p_big[0])  # (3,H,W)

    return img_t[0], torch.stack(vis, dim=0), (gx, gy)


class GlimpseViewerTk:
    def __init__(self, img_path: str, Config, init_xy: Optional[Tuple[float, float]] = None):
        self.Config = Config
        self.img_path = img_path
        self.root = tk.Tk()
        self.root.title("Glimpse Viewer")

        # Display scales
        self.full_scale = int(getattr(self.Config, 'VIEWER_FULL_SCALE', 8))   # e.g., 60x60 -> 240x240
        self.glimpse_scale = int(getattr(self.Config, 'VIEWER_GLI_SCALE', 64))  # e.g., 4x4 -> 256x256

        self.full_img_t = None  # (3,H,W) [-1,1]
        self.gxgy = (0.5, 0.5)
        if init_xy is not None:
            self.gxgy = (float(init_xy[0]), float(init_xy[1]))

        # UI layout
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 8))
        self.lbl_info = ttk.Label(top, text="Click on the image to set gaze position")
        self.lbl_info.pack(side=tk.LEFT)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, padx=(0, 8))
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Full image label (clickable)
        self.full_label = ttk.Label(left)
        self.full_label.pack()
        self.full_label.bind("<Button-1>", self.on_click)

        # Glimpse labels (created dynamically)
        self.glimpse_frame = ttk.Frame(right)
        self.glimpse_frame.pack()
        self.glimpse_labels: list[ttk.Label] = []

        # Keep PhotoImage refs
        self._photo_full = None
        self._photo_glimpses = []

        # Build first view
        self.load_full_image()
        self.update_glimpses()

    def load_full_image(self):
        transform = transforms.Compose([
            transforms.Resize(self.Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        img = Image.open(self.img_path).convert('RGB')
        img_t = transform(img)  # (3,H,W)
        self.full_img_t = img_t

        # Draw full image with a cross at current gaze
        H, W = self.Config.IMG_SIZE
        img01 = to01(img_t).permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray((img01 * 255).astype(np.uint8))
        # Upscale full image for display
        disp_size = (W * self.full_scale, H * self.full_scale)
        pil = pil.resize(disp_size, Image.NEAREST)
        # draw crosshair by simple pixels around location in scaled coords
        cx0 = int(self.gxgy[0] * (W - 1)) * self.full_scale
        cy0 = int(self.gxgy[1] * (H - 1)) * self.full_scale
        for dx in range(-2 * self.full_scale, 2 * self.full_scale + 1):
            x = min(max(cx0 + dx, 0), disp_size[0] - 1)
            pil.putpixel((x, cy0), (255, 0, 0))
        for dy in range(-2 * self.full_scale, 2 * self.full_scale + 1):
            y = min(max(cy0 + dy, 0), disp_size[1] - 1)
            pil.putpixel((cx0, y), (255, 0, 0))
        self._photo_full = ImageTk.PhotoImage(pil)
        self.full_label.configure(image=self._photo_full)
        self.full_label.image = self._photo_full
        self.lbl_info.configure(text=f"Gaze (norm): x={self.gxgy[0]:.3f}, y={self.gxgy[1]:.3f}")

    def update_glimpses(self):
        # Build glimpses given current gxgy
        img0, vis_stack, _ = build_glimpses(self.img_path, self.Config, self.gxgy, vis_scale=self.glimpse_scale)

        # Ensure labels match K_SCALES
        k = vis_stack.size(0)
        while len(self.glimpse_labels) < k:
            lbl = ttk.Label(self.glimpse_frame)
            lbl.pack(side=tk.LEFT, padx=4)
            self.glimpse_labels.append(lbl)
        for i in range(len(self.glimpse_labels)):
            if i >= k:
                self.glimpse_labels[i].configure(image='')
                self.glimpse_labels[i].image = None

        # Convert each glimpse to PhotoImage and update
        self._photo_glimpses = []
        for i in range(k):
            g = vis_stack[i].permute(1, 2, 0).cpu().numpy()  # HxWx3 in [0,1]
            pil = Image.fromarray((g * 255).astype(np.uint8))
            ph = ImageTk.PhotoImage(pil)
            self._photo_glimpses.append(ph)
            self.glimpse_labels[i].configure(image=ph)
            self.glimpse_labels[i].image = ph

    def on_click(self, event):
        # event.x/y are in label coordinates; label image is exact IMG_SIZE
        H, W = self.Config.IMG_SIZE
        # Convert scaled coords back to original pixel grid
        x = float(np.clip(event.x, 0, W * self.full_scale - 1))
        y = float(np.clip(event.y, 0, H * self.full_scale - 1))
        px = int(round(x / self.full_scale))
        py = int(round(y / self.full_scale))
        px = int(np.clip(px, 0, W - 1))
        py = int(np.clip(py, 0, H - 1))
        gx = px / (W - 1 if W > 1 else 1)
        gy = py / (H - 1 if H > 1 else 1)
        self.gxgy = (gx, gy)
        self.load_full_image()
        self.update_glimpses()

    def run(self):
        self.root.mainloop()


# (Matplotlib interactive helper removed; Tk GUI is used instead)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize encoder glimpses at a selected position (Tk GUI)')
    parser.add_argument('image', help='Path to an input image file')
    parser.add_argument('--config', default='config_maze', help='Config module to import (default: config_maze)')
    parser.add_argument('--x', type=float, default=None, help='Gaze x (normalized [0,1] or pixels if >1)')
    parser.add_argument('--y', type=float, default=None, help='Gaze y (normalized [0,1] or pixels if >1)')

    args = parser.parse_args()

    # Load Config from module
    try:
        mod = importlib.import_module(args.config)
        Config = mod.Config
        print(f"Loaded Config from {args.config}")
    except Exception as e:
        print(f"Failed to import config module '{args.config}': {e}")
        raise SystemExit(1)

    init_xy = None
    if args.x is not None and args.y is not None:
        gx, gy = float(args.x), float(args.y)
        if gx > 1.0 or gy > 1.0:
            W = Config.IMG_SIZE[1]
            H = Config.IMG_SIZE[0]
            gx = np.clip(gx / (W - 1), 0.0, 1.0)
            gy = np.clip(gy / (H - 1), 0.0, 1.0)
        init_xy = (gx, gy)

    app = GlimpseViewerTk(args.image, Config, init_xy)
    app.run()
