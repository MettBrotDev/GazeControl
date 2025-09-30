import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import importlib

from models import GazeControlModel
from test_model import load_random_image

# Global variables
Config = None
model = None
device = None
current_image = None
current_gaze = None
memory = None
fig = None
axes = None
canvas = None


def crop_patch(image, gaze, crop_size, resize_to=None):
    """Crop a patch around the gaze from a batched image tensor.
    image: (B,3,H,W) in [0,1]
    gaze: (B,2) normalized [0,1]
    crop_size: (h,w)
    resize_to: (h,w) optional final size
    returns (B,3,h,w)
    """
    B, C, H, W = image.shape
    ch, cw = crop_size
    cx = int(gaze[0, 0].item() * (W - 1))
    cy = int(gaze[0, 1].item() * (H - 1))
    x0 = cx - cw // 2
    y0 = cy - ch // 2
    x1 = x0 + cw
    y1 = y0 + ch

    x0_src = max(0, min(W, x0))
    y0_src = max(0, min(H, y0))
    x1_src = max(0, min(W, x1))
    y1_src = max(0, min(H, y1))

    crop = image[:, :, y0_src:y1_src, x0_src:x1_src]

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        # Match training behavior: zero pad out-of-bounds regions
        crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    if crop.shape[-2:] != (ch, cw):
        crop = F.interpolate(crop, size=(ch, cw), mode='bilinear', align_corners=False)

    if resize_to is not None and (resize_to[0] != ch or resize_to[1] != cw):
        crop = F.interpolate(crop, size=resize_to, mode='bilinear', align_corners=False)
    return crop


class ManualGazeTestGUI:
    def __init__(self, root, config_module, checkpoint_path, source):
        self.root = root
        self.root.title("Manual Gaze Control Test")
        
        # Load config and model
        self.load_config_and_model(config_module, checkpoint_path)
        
        # Set up GUI
        self.setup_gui()
        
        # Load initial image
        self.load_new_image(source)
        
    def load_config_and_model(self, config_module, checkpoint_path):
        global Config, model, device
        
        # Import config
        config_mod = importlib.import_module(config_module)
        Config = config_mod.Config
        
        device = Config.DEVICE
        
        # Load model (new signature, no memory args)
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
        
        # Handle PastRuns folder for checkpoint
        if not os.path.isabs(checkpoint_path):
            past_runs_dir = os.path.join(os.path.dirname(__file__), "PastRuns")
            checkpoint_path = os.path.join(past_runs_dir, checkpoint_path)
        
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        
    def setup_gui(self):
        global fig, axes, canvas
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="New Image", command=self.new_image_clicked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Reset Gaze", command=self.reset_gaze).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Random Step", command=self.random_step).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Click on the original image to set gaze position")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Manual Gaze Control Test")
        
        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(fig, main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click event
        canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_new_image(self, source):
        global current_image, current_gaze, memory
        
        current_image = load_random_image(source, Config)
        self.reset_gaze()
        self.update_display()
        
    def reset_gaze(self):
        global current_gaze, memory
        
        # Start at center
        frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1)) if getattr(Config, 'USE_GAZE_BOUNDS', False) else 0.0
        lo, hi = frac, 1.0 - frac
        current_gaze = torch.tensor([[0.5, 0.5]], device=device, dtype=torch.float32)
        if getattr(Config, 'USE_GAZE_BOUNDS', False):
            current_gaze = current_gaze.clamp(min=lo, max=hi)
        
        # Reset LSTM memory
        memory = model.init_memory(1, device)
        
        self.status_label.config(text="Gaze reset to center. Click to set new position.")
        self.update_display()
        
    def new_image_clicked(self):
        # Use the same source as initially loaded (you might want to make this configurable)
        source = getattr(self, 'current_source', 'local')
        self.load_new_image(source)
        
    def on_click(self, event):
        global current_gaze, memory
        
        # Only respond to clicks on the original image (top-left subplot)
        if event.inaxes != axes[0, 0]:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        # Use displayed tensor shape for robustness
        _, _, H, W = current_image.shape
        gaze_x = event.xdata / W
        gaze_y = event.ydata / H
        
        # Clamp to valid range
        gaze_x = max(0, min(1, gaze_x))
        gaze_y = max(0, min(1, gaze_y))
        
        current_gaze = torch.tensor([[gaze_x, gaze_y]], device=device, dtype=torch.float32)
        if getattr(Config, 'USE_GAZE_BOUNDS', False):
            frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
            lo, hi = frac, 1.0 - frac
            current_gaze = current_gaze.clamp(min=lo, max=hi)
        
        # Process this gaze position through the model
        self.process_gaze_step()
        
    def random_step(self):
        """Move gaze by a random delta and process a step (matches training)."""
        global current_gaze
        delta = (torch.rand_like(current_gaze) * 2.0 - 1.0) * Config.MAX_MOVE
        current_gaze = torch.clamp(current_gaze + delta, 0.0, 1.0)
        if getattr(Config, 'USE_GAZE_BOUNDS', False):
            frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
            lo, hi = frac, 1.0 - frac
            current_gaze = current_gaze.clamp(min=lo, max=hi)
        self.process_gaze_step()
        
    def process_gaze_step(self):
        global current_gaze, memory
        
        # Prepare unblurred cutout around gaze
        # Build multi-scale patches (match training)
        base_h, base_w = Config.FOVEA_CROP_SIZE
        k_scales = int(getattr(Config, 'K_SCALES', 3))
        patches = []
        for i in range(k_scales):
            scale = 2 ** i
            size = (base_h * scale, base_w * scale)
            patches.append(crop_patch(current_image, current_gaze, size, resize_to=Config.FOVEA_OUTPUT_SIZE).to(device))
        patch = patches[0]
        
        # Track memory change
        h_prev, c_prev = memory[0].clone().detach(), memory[1].clone().detach()
        
        # Get model prediction (forward expects a list/tuple of patches)
        with torch.no_grad():
            rec, memory = model(patches, memory, current_gaze)
        
        # Compute memory change metric
        h_diff = (memory[0] - h_prev).abs()
        c_diff = (memory[1] - c_prev).abs()
        mem_change_text = f"h mean {h_diff.mean().item():.4f}, c mean {c_diff.mean().item():.4f}"
        
        # Suggest next random gaze (for visualization)
        delta = (torch.rand_like(current_gaze) * 2.0 - 1.0) * Config.MAX_MOVE
        predicted_next_gaze = torch.clamp(current_gaze + delta, 0.0, 1.0)
        if getattr(Config, 'USE_GAZE_BOUNDS', False):
            frac = float(getattr(Config, 'GAZE_BOUND_FRACTION', 0.1))
            lo, hi = frac, 1.0 - frac
            predicted_next_gaze = predicted_next_gaze.clamp(min=lo, max=hi)

        # Build composite centered overlay and update display
        # Put largest patch first for correct background -> mid -> small overlay
        composite = self._composite_centered([p.cpu()[0] for p in reversed(patches)])
        self.update_display(patch.cpu()[0], rec.cpu()[0], predicted_gaze=predicted_next_gaze, memory_change=mem_change_text, composite=composite)

        # Update status
        self.status_label.config(text=f"Gaze: ({current_gaze[0,0]:.3f}, {current_gaze[0,1]:.3f}) | Mem Δ: {mem_change_text}")
        
    def _composite_centered(self, patches, base_upscale: int = 4):
        if patches is None or len(patches) == 0:
            return None
        # Canvas = 9 * base_upscale (e.g., 36), overlays = [1.0, 0.5, 0.25] of canvas
        canvas_size = int(9 * base_upscale)
        canvas = torch.zeros(3, canvas_size, canvas_size)
        scales = [1.0, 0.5, 0.25]
        colors = [
            torch.tensor([1.0, 1.0, 1.0]),  # large - white
            torch.tensor([1.0, 1.0, 0.0]),  # mid - yellow
            torch.tensor([1.0, 0.0, 0.0]),  # small - red
        ]
        outline_stride = 2  # dashed to look thinner
        outline_alpha = 0.6
        for i, p in enumerate(patches[:3]):
            target = max(1, int(round(canvas_size * scales[i])))
            p_res = F.interpolate(p.unsqueeze(0), size=(target, target), mode='nearest')[0]
            y0 = (canvas_size - target) // 2
            x0 = (canvas_size - target) // 2
            canvas[:, y0:y0+target, x0:x0+target] = p_res
            # Draw dashed 1px outline with alpha blending for thinner look
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

    def update_display(self, foveal_view=None, reconstruction=None, predicted_gaze=None, memory_change=None, composite=None):
        global current_gaze
        
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
            ax.axis('off')
        
        # Original image with current gaze position
        img_np = current_image[0].permute(1, 2, 0).cpu().numpy()
        H, W = img_np.shape[:2]
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image (Click to set gaze)")
        
        # Mark current gaze position
        if current_gaze is not None:
            gx = current_gaze[0, 0].cpu().item() * W
            gy = current_gaze[0, 1].cpu().item() * H
            axes[0, 0].scatter(gx, gy, c='red', marker='x', s=200, linewidths=3, label='Current Gaze')
        
        # Mark predicted next gaze position
        if predicted_gaze is not None:
            px = predicted_gaze[0, 0].cpu().item() * W
            py = predicted_gaze[0, 1].cpu().item() * H
            axes[0, 0].scatter(px, py, c='blue', marker='o', s=100, alpha=0.7, label='Next (rand)')
            if current_gaze is not None:
                axes[0, 0].annotate('', xy=(px, py), xytext=(gx, gy),
                                   arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.7))
        
        axes[0, 0].legend()
        
        # Foveal view (cutout patch)
        if foveal_view is not None:
            axes[0, 1].imshow(foveal_view.permute(1, 2, 0).numpy())
            axes[0, 1].set_title("Cutout Patch")
        else:
            axes[0, 1].set_title("Cutout Patch (click to generate)")
            
        # Composite multi-scale view (bottom-right)
        if composite is not None:
            axes[1, 1].imshow(composite)
            axes[1, 1].set_title("Composite (multi-scale)")
        else:
            axes[1, 1].set_title("Composite (generate a step)")

        # Reconstruction
        if reconstruction is not None:
            axes[1, 0].imshow(reconstruction.permute(1, 2, 0).numpy())
            axes[1, 0].set_title("Reconstruction (step)")
        else:
            axes[1, 0].set_title("Reconstruction (click to generate)")
            
        # Minimal info text on top-right
        axes[0, 1].text(0.05, 1.02, "Click original to set gaze", fontsize=10, transform=axes[0, 1].transAxes)
        if memory_change is not None:
            axes[0, 1].text(0.05, 0.97, f"Memory Δ: {memory_change}", fontsize=10, color='purple', transform=axes[0, 1].transAxes)
        
        # Refresh canvas
        canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", default="config",
                        help="config module name (e.g. 'config' or 'configSmol')")
    parser.add_argument("--checkpoint", required=True, help="path to .pth")
    parser.add_argument("--source", choices=["mnist", "local", "cifar100"], default=None,
                        help="image source (defaults to config's DATA_SOURCE)")
    args = parser.parse_args()
    
    # Import config first to get default source
    config_mod = importlib.import_module(args.config_module)
    default_source = args.source or config_mod.Config.DATA_SOURCE
    
    # Create GUI
    root = tk.Tk()
    app = ManualGazeTestGUI(root, args.config_module, args.checkpoint, default_source)
    
    # Store source for new image button
    app.current_source = default_source
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()
