import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import time

class FoveatedBlur:
    def __init__(self, image_size=272, output_size=224, center_patch_size=1, max_patch_size=32, eccentricity_factor=2.0):
        """
        Args:
            image_size: Size of the square input image
            output_size: Size of the square output image (consistent across all foveations)
            center_patch_size: Patch size at the center of attention (usually 1)
            max_patch_size: Maximum patch size at the periphery
            eccentricity_factor: Controls how quickly patch size increases with distance
        """
        self.image_size = image_size
        self.output_size = output_size
        self.center_patch_size = center_patch_size
        self.max_patch_size = max_patch_size
        self.eccentricity_factor = eccentricity_factor
        
    def get_patch_size(self, distance_from_center):
        """Calculate patch size based on distance from center of attention."""
        # Use a smoother exponential growth
        normalized_distance = min(distance_from_center, 1.0)  # Cap at 1.0
        patch_size = self.center_patch_size + (self.max_patch_size - self.center_patch_size) * (normalized_distance ** self.eccentricity_factor)
        return max(self.center_patch_size, int(patch_size))
    
    def apply_foveated_blur(self, image, center_x, center_y):
        """
        Fast multi‐scale foveated blur. Returns a blurred image
        of shape [B, C, output_size, output_size].
        """
        B, C, H, W = image.shape
        half = self.output_size // 2
        cx, cy = int(center_x * W), int(center_y * H)

        # 1) crop & pad to output_size
        top, bottom = max(0, cy-half), min(H, cy+half)
        left, right = max(0, cx-half), min(W, cx+half)
        pad_t, pad_b = max(0, half-cy), max(0, cy+half-H)
        pad_l, pad_r = max(0, half-cx), max(0, cx+half-W)
        roi = image[:, :, top:bottom, left:right]
        if pad_t or pad_b or pad_l or pad_r:
            roi = F.pad(roi, (pad_l, pad_r, pad_t, pad_b), value=0)
        roi = roi[:, :, :self.output_size, :self.output_size]

        # 2) build radial distance map normalized to [0,1]
        device = roi.device
        ys = torch.arange(self.output_size, device=device).float() - half
        xs = torch.arange(self.output_size, device=device).float() - half
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        D = torch.sqrt(Y**2 + X**2) / (np.sqrt(2) * half)

        # 3) compute per‐pixel patch sizes
        raw_ps = (self.center_patch_size +
                  (self.max_patch_size - self.center_patch_size) *
                  D.clamp(0,1).pow(self.eccentricity_factor))
        ps = raw_ps.floor().clamp(self.center_patch_size, self.max_patch_size).int()

        # 4) blur & blend by mask
        fov = torch.zeros_like(roi)
        for k in ps.unique():
            k = int(k)
            if k <= 1:
                blurred = roi
            else:
                blurred = F.avg_pool2d(roi, kernel_size=k, stride=1, padding=k//2)
            
            # Crop back to desired output_size if padding increased size
            if blurred.shape[2] > self.output_size or blurred.shape[3] > self.output_size:
                blurred = blurred[:, :, :self.output_size, :self.output_size]

            mask = (ps == k).float().to(device)  # [H,W]
            fov += blurred * mask.unsqueeze(0).unsqueeze(0)

        return fov, ps

def test_foveated_blur():
    """Test function to create and visualize foveated blur effect."""
    
    # Create a test image with some patterns
    def create_test_image():
        # Create a 272x272 RGB image with some patterns
        img = torch.zeros(3, 272, 272)
        
        # Add some circles and patterns
        y, x = torch.meshgrid(torch.arange(272), torch.arange(272), indexing='ij')
        
        # Checkerboard pattern
        checker = ((x // 16) + (y // 16)) % 2
        img[0] = checker * 0.8  # Red channel
        
        # Circular pattern
        center = 136
        radius = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        circle = (torch.sin(radius / 10) + 1) / 2
        img[1] = circle * 0.7  # Green channel
        
        # Diagonal stripes
        diagonal = torch.sin((x + y) / 8)
        img[2] = (diagonal + 1) / 2 * 0.6  # Blue channel
        
        return img
    
    # Create test image
    test_img = create_test_image().unsqueeze(0)  # Add batch dimension
    
    # These parameters look appropriate
    blur = FoveatedBlur(image_size=272, output_size=182, center_patch_size=1, max_patch_size=48, eccentricity_factor=2)
    
    # Test different center positions
    centers = [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]
    
    # Set matplotlib backend for non-interactive environments
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    fig, axes = plt.subplots(len(centers), 3, figsize=(12, 4*len(centers)))
    if len(centers) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (center_x, center_y) in enumerate(centers):
        # Apply foveated blur (now returns foveated image and patch map)
        foveated_img, patch_map_tensor = blur.apply_foveated_blur(test_img, center_x, center_y)
        
        patch_map = patch_map_tensor.cpu()
        
        # Plot results
        # Original image
        axes[i, 0].imshow(test_img[0].permute(1, 2, 0))
        axes[i, 0].set_title(f'Original Image (272x272)')
        axes[i, 0].axis('off')
        
        # Add center point
        center_px_x = int(center_x * 272)
        center_px_y = int(center_y * 272)
        axes[i, 0].plot(center_px_x, center_px_y, 'r+', markersize=20, markeredgewidth=3)
        
        # Foveated image
        axes[i, 1].imshow(foveated_img[0].permute(1, 2, 0).cpu().numpy(), interpolation='none')
        axes[i, 1].set_title(f'Foveated Image ({blur.output_size}x{blur.output_size})\nCenter: ({center_x:.1f}, {center_y:.1f})')
        axes[i, 1].axis('off')
        # Center is always at middle of output
        center_out = blur.output_size // 2
        axes[i, 1].plot(center_out, center_out, 'r+', markersize=20, markeredgewidth=3)
        
        # Patch size map
        im = axes[i, 2].imshow(patch_map, cmap='viridis', interpolation='none')
        axes[i, 2].set_title(f'Blur Level Map ({blur.output_size}x{blur.output_size})')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2])
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig('/home/fdoll/GazeControl/foveated_blur_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'foveated_blur_test.png'")
    plt.close()  # Close the figure to free memory
    
    # Get stats and timing for both CPU and GPU. The stats below will be from the last run.
    print("\n--- Performance Test ---")
    
    # CPU Performance
    cpu_img = test_img.to('cpu')
    start_time = time.perf_counter()
    foveated_img, patch_map_tensor = blur.apply_foveated_blur(cpu_img, centers[-1][0], centers[-1][1])
    end_time = time.perf_counter()
    blur_time_ms = (end_time - start_time) * 1000
    print(f"Blurring took: {blur_time_ms:.2f} ms (on CPU)")

    # GPU Performance
    if torch.cuda.is_available():
        gpu_img = test_img.to('cuda')
        # Warm-up run for GPU to handle kernel loading etc.
        blur.apply_foveated_blur(gpu_img, centers[-1][0], centers[-1][1])
        torch.cuda.synchronize()
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        foveated_img, patch_map_tensor = blur.apply_foveated_blur(gpu_img, centers[-1][0], centers[-1][1])
        ender.record()
        torch.cuda.synchronize()
        blur_time_ms = starter.elapsed_time(ender)
        print(f"Blurring took: {blur_time_ms:.2f} ms (on GPU)")
    else:
        print("CUDA not available, skipping GPU performance test.")

    patch_map = patch_map_tensor.cpu()

    # Print some statistics
    print("\nFoveated Blur Test Results:")
    print(f"Input image size: {test_img.shape}")
    print(f"Output foveated image shape: {foveated_img.shape}")
    
    # Show patch size distribution and calculate compression
    patch_sizes_np = patch_map.numpy()
    unique_sizes, counts = np.unique(patch_sizes_np, return_counts=True)
    print(f"\nBlur level distribution (pixels per level):")
    
    effective_pixels = 0
    for size, count in zip(unique_sizes, counts):
        print(f"  Level {size} (equiv. {size}x{size} patch): {count:,} pixels")
        effective_pixels += count / (size * size)

    total_pixels = blur.output_size * blur.output_size
    compression_ratio = total_pixels / effective_pixels if effective_pixels > 0 else 0
    
    print(f"\nTotal pixels in output: {total_pixels:,}")
    print(f"Effective pixels (information content): {int(effective_pixels):,}")
    print(f"Compression ratio: {compression_ratio:.1f}x")


if __name__ == "__main__":
    print("Testing foveated blur functionality...")
    test_foveated_blur()