import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import importlib

from models import GazeControlModel
from foveal_blur import foveal_blur
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
        
        # Load model
        model = GazeControlModel(
            encoder_output_size=Config.ENCODER_OUTPUT_SIZE,
            state_size=Config.HIDDEN_SIZE,
            img_size=Config.IMG_SIZE,
            fovea_size=Config.FOVEA_OUTPUT_SIZE,
            memory_size=Config.MEMORY_SIZE,
            memory_dim=getattr(Config, 'MEMORY_DIM', 4)
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
        current_gaze = torch.tensor([[0.5, 0.5]], device=device, dtype=torch.float32)
        
        # Reset spatial memory
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
            
        # Convert click coordinates to normalized gaze position
        img_height, img_width = Config.IMG_SIZE
        gaze_x = event.xdata / img_width
        gaze_y = event.ydata / img_height
        
        # Clamp to valid range
        gaze_x = max(0, min(1, gaze_x))
        gaze_y = max(0, min(1, gaze_y))
        
        current_gaze = torch.tensor([[gaze_x, gaze_y]], device=device, dtype=torch.float32)
        
        # Process this gaze position through the model
        self.process_gaze_step()
        
    def process_gaze_step(self):
        global current_gaze, memory
        
        # Get current image as numpy array
        img_np = (current_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Get foveal view
        cx = int(current_gaze[0, 0] * Config.IMG_SIZE[0])
        cy = int(current_gaze[0, 1] * Config.IMG_SIZE[1])
        
        blurred = foveal_blur(img_np, (cx, cy),
                             output_size=Config.FOVEA_OUTPUT_SIZE,
                             crop_size=Config.FOVEA_CROP_SIZE)
        
        fov = torch.from_numpy(blurred.astype(np.float32) / 255.0)\
                   .permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Track memory change
        prev_memory = memory.clone().detach()
        
        # Get model prediction
        with torch.no_grad():
            rec, (delta, stop), _, memory = model.sample_action(fov, memory, current_gaze)
        
        # Compute memory change metric (percent of elements changed > 1e-3 and mean abs change)
        diff = (memory - prev_memory).abs()
        mean_change = diff.mean().item()
        percent_changed = (diff > 1e-3).float().mean().item() * 100.0
        mem_change_text = f"{percent_changed:.2f}% (>1e-3), mean {mean_change:.4f}"
        
        # Calculate predicted next gaze
        delta_clamped = delta.clamp(-Config.MAX_MOVE, Config.MAX_MOVE)
        predicted_next_gaze = (current_gaze + delta_clamped).clamp(0, 1)
        
        # Update display
        self.update_display(fov.cpu()[0], rec.cpu()[0], predicted_gaze=predicted_next_gaze, stop_signal=stop.item(), memory_change=mem_change_text)
        
        # Update status
        stop_text = "STOP" if stop.item() == 1 else "CONTINUE"
        self.status_label.config(text=f"Gaze: ({current_gaze[0,0]:.3f}, {current_gaze[0,1]:.3f}) | Prediction: {stop_text} | Mem Δ: {mem_change_text}")
        
    def update_display(self, foveal_view=None, reconstruction=None, predicted_gaze=None, stop_signal=None, memory_change=None):
        global current_gaze
        
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
            ax.axis('off')
        
        # Original image with current gaze position
        axes[0, 0].imshow(current_image[0].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title("Original Image (Click to set gaze)")
        
        # Mark current gaze position
        if current_gaze is not None:
            gx = current_gaze[0, 0].cpu().item() * Config.IMG_SIZE[0]
            gy = current_gaze[0, 1].cpu().item() * Config.IMG_SIZE[1]
            axes[0, 0].scatter(gx, gy, c='red', marker='x', s=200, linewidths=3, label='Current Gaze')
        
        # Mark predicted next gaze position
        if predicted_gaze is not None:
            px = predicted_gaze[0, 0].cpu().item() * Config.IMG_SIZE[0]
            py = predicted_gaze[0, 1].cpu().item() * Config.IMG_SIZE[1]
            axes[0, 0].scatter(px, py, c='blue', marker='o', s=100, alpha=0.7, label='Predicted Next')
            
            # Draw arrow from current to predicted
            if current_gaze is not None:
                axes[0, 0].annotate('', xy=(px, py), xytext=(gx, gy),
                                   arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.7))
        
        axes[0, 0].legend()
        
        # Foveal view
        if foveal_view is not None:
            axes[0, 1].imshow(foveal_view.permute(1, 2, 0).numpy())
            axes[0, 1].set_title("Foveal View")
        else:
            axes[0, 1].set_title("Foveal View (click to generate)")
            
        # Reconstruction
        if reconstruction is not None:
            axes[1, 0].imshow(reconstruction.permute(1, 2, 0).numpy())
            axes[1, 0].set_title("Reconstruction")
        else:
            axes[1, 0].set_title("Reconstruction (click to generate)")
            
        # Info panel
        axes[1, 1].text(0.1, 0.8, "Instructions:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, "• Click on original image to set gaze", fontsize=10)
        axes[1, 1].text(0.1, 0.6, "• Red X = current gaze position", fontsize=10)
        axes[1, 1].text(0.1, 0.5, "• Blue O = model's predicted next gaze", fontsize=10)
        axes[1, 1].text(0.1, 0.4, "• Arrow shows predicted movement", fontsize=10)
        axes[1, 1].text(0.1, 0.3, "• Use 'New Image' for different image", fontsize=10)
        axes[1, 1].text(0.1, 0.2, "• Use 'Reset Gaze' to start over", fontsize=10)
        
        # Memory change metric
        if memory_change is not None:
            axes[1, 1].text(0.1, 0.1, f"Memory change: {memory_change}", fontsize=10, color='purple')
        
        if stop_signal is not None:
            stop_text = "Model says: STOP" if stop_signal == 1 else "Model says: CONTINUE"
            color = 'red' if stop_signal == 1 else 'green'
            axes[1, 1].text(0.1, 0.05, stop_text, fontsize=12, fontweight='bold', color=color)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
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
