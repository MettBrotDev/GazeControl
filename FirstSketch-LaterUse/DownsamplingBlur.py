import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import time

class FoveatedCartesianGrid:
    def __init__(self, image_size=272, output_grid_size=59, center_patch_size=1, max_patch_size=32, eccentricity_factor=2.0):
        """
        Foveated Cartesian Grid (FCG) Sampling for CNN preprocessing.
        
        Args:
            image_size: Size of the square input image
            output_grid_size: Size of the compact output grid (e.g., 59x59)
            center_patch_size: Patch size at the center of attention (usually 1)
            max_patch_size: Maximum patch size at the periphery
            eccentricity_factor: Controls how quickly patch size increases with distance
        """
        self.image_size = image_size
        self.output_grid_size = output_grid_size
        self.center_patch_size = center_patch_size
        self.max_patch_size = max_patch_size
        self.eccentricity_factor = eccentricity_factor
        
        # Pre-compute sampling grid coordinates
        self._precompute_sampling_grid()
    
    def _precompute_sampling_grid(self):
        """Pre-compute the sampling grid coordinates and patch sizes."""
        # Create output grid coordinates
        grid_coords = torch.arange(self.output_grid_size).float()
        grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        
        # Center the grid coordinates
        center = self.output_grid_size // 2
        grid_y = grid_y - center
        grid_x = grid_x - center
        
        # Calculate distance from center (normalized)
        max_distance = np.sqrt(2) * center
        distance = torch.sqrt(grid_y**2 + grid_x**2) / max_distance
        distance = distance.clamp(0, 1)
        
        # Calculate patch sizes based on distance
        patch_sizes = (self.center_patch_size + 
                      (self.max_patch_size - self.center_patch_size) * 
                      distance.pow(self.eccentricity_factor))
        self.patch_sizes = patch_sizes.floor().clamp(self.center_patch_size, self.max_patch_size).int()
        
        # Store grid coordinates for sampling
        self.grid_y = grid_y
        self.grid_x = grid_x
        self.distance_map = distance
    
    def sample_foveated_patches(self, image, center_x, center_y):
        """
        Sample patches from the image based on foveated pattern.
        
        Args:
            image: Input image tensor [B, C, H, W]
            center_x, center_y: Attention center coordinates (0-1 range)
            
        Returns:
            foveated_grid: Compact grid ready for CNN [B, C, output_grid_size, output_grid_size]
        """
        B, C, H, W = image.shape
        device = image.device
        
        # Convert center coordinates to pixel space
        cx = int(center_x * W)
        cy = int(center_y * H)
        
        # Move grids to device
        patch_sizes = self.patch_sizes.to(device)
        grid_y = self.grid_y.to(device)
        grid_x = self.grid_x.to(device)
        
        # Initialize output grid
        foveated_grid = torch.zeros(B, C, self.output_grid_size, self.output_grid_size, device=device)
        
        # Sample patches for each position in the output grid
        for i in range(self.output_grid_size):
            for j in range(self.output_grid_size):
                patch_size = patch_sizes[i, j].item()
                
                # Calculate source coordinates in the original image
                # Map grid position to image coordinates
                scale_factor = self.image_size / self.output_grid_size
                src_y = cy + int(grid_y[i, j] * scale_factor)
                src_x = cx + int(grid_x[i, j] * scale_factor)
                
                # Extract patch from source image
                half_patch = patch_size // 2
                y1 = max(0, src_y - half_patch)
                y2 = min(H, src_y + half_patch + (patch_size % 2))
                x1 = max(0, src_x - half_patch)
                x2 = min(W, src_x + half_patch + (patch_size % 2))
                
                if y2 > y1 and x2 > x1:
                    patch = image[:, :, y1:y2, x1:x2]
                    
                    # Average pool the patch to get single pixel value
                    if patch.numel() > 0:
                        pooled_patch = F.adaptive_avg_pool2d(patch, (1, 1))
                        foveated_grid[:, :, i, j] = pooled_patch.squeeze(-1).squeeze(-1)
        
        return foveated_grid
    
    def get_compression_stats(self):
        """Calculate compression statistics."""
        total_input_pixels = self.image_size * self.image_size
        total_output_pixels = self.output_grid_size * self.output_grid_size
        
        spatial_compression = total_input_pixels / total_output_pixels
        
        return {
            'input_pixels': total_input_pixels,
            'output_pixels': total_output_pixels,
            'spatial_compression': spatial_compression,
            'size_reduction': (1 - total_output_pixels / total_input_pixels) * 100
        }

def test_foveated_downsampling():
    """Test function to create and visualize foveated downsampling effect."""
    
    # Create a test image with some patterns (same as ImgBlur.py)
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
    
    # Initialize foveated downsampler
    fcg = FoveatedCartesianGrid(
        image_size=272, 
        output_grid_size=59, 
        center_patch_size=1, 
        max_patch_size=32, 
        eccentricity_factor=2.0
    )
    
    # Test different center positions (same as ImgBlur.py)
    centers_visual = [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]
    centers_timing = [(0.1, 0.1), (0.2, 0.3), (0.3, 0.7), (0.4, 0.2), (0.5, 0.5), 
                     (0.6, 0.8), (0.7, 0.4), (0.8, 0.7), (0.9, 0.6), (0.5, 0.1),
                     (0.1, 0.5), (0.9, 0.9), (0.3, 0.3), (0.7, 0.7), (0.2, 0.8)]
    
    # Storage for timing measurements
    compression_times = []
    
    # Set matplotlib backend for non-interactive environments
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(len(centers_visual), 4, figsize=(16, 4*len(centers_visual)))
    if len(centers_visual) == 1:
        axes = axes.reshape(1, -1)
    
    # First run timing tests on all centers
    print("Running timing tests on multiple center positions...")
    for center_x, center_y in centers_timing:
        start_time = time.time()
        
        foveated_grid = fcg.sample_foveated_patches(
            test_img, center_x, center_y
        )
        
        if torch.cuda.is_available() and foveated_grid.is_cuda:
            torch.cuda.synchronize()
        
        end_time = time.time()
        compression_time = (end_time - start_time) * 1000  # Convert to milliseconds
        compression_times.append(compression_time)
    
    # Then create visualizations for just the visual centers
    for i, (center_x, center_y) in enumerate(centers_visual):
        # Apply foveated downsampling
        foveated_grid = fcg.sample_foveated_patches(
            test_img, center_x, center_y
        )
        
        # Ensure GPU operations complete if using CUDA
        if torch.cuda.is_available() and foveated_grid.is_cuda:
            torch.cuda.synchronize()
        
        # Convert to numpy for plotting
        rgb_grid = foveated_grid[0].permute(1, 2, 0).cpu().numpy()
        patch_sizes = fcg.patch_sizes.cpu().numpy()
        
        # Plot results
        # Original image
        axes[i, 0].imshow(test_img[0].permute(1, 2, 0))
        axes[i, 0].set_title(f'Original Image (272x272)')
        axes[i, 0].axis('off')
        
        # Add center point
        center_px_x = int(center_x * 272)
        center_px_y = int(center_y * 272)
        axes[i, 0].plot(center_px_x, center_px_y, 'r+', markersize=20, markeredgewidth=3)
        
        # Foveated grid
        axes[i, 1].imshow(rgb_grid, interpolation='none')
        axes[i, 1].set_title(f'Foveated Grid (59x59)\nCenter: ({center_x:.1f}, {center_y:.1f})')
        axes[i, 1].axis('off')
        # Center is always at middle of output
        center_out = fcg.output_grid_size // 2
        axes[i, 1].plot(center_out, center_out, 'r+', markersize=10, markeredgewidth=2)
        
        # Patch size map
        im2 = axes[i, 2].imshow(patch_sizes, cmap='viridis', interpolation='none')
        axes[i, 2].set_title(f'Patch Size Map (59x59)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('./foveated_downsampling_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'foveated_downsampling_test.png'")
    plt.close()
    
    # Print compression statistics
    stats = fcg.get_compression_stats()
    print("\nFoveated Downsampling Test Results:")
    print(f"Input image size: {test_img.shape}")
    print(f"Output grid shape: {foveated_grid.shape}")
    print(f"  - RGB channels: {foveated_grid.shape[1]}")
    
    print(f"\nCompression Statistics:")
    print(f"Input pixels: {stats['input_pixels']:,}")
    print(f"Output pixels: {stats['output_pixels']:,}")
    print(f"Spatial compression ratio: {stats['spatial_compression']:.1f}x")
    print(f"Size reduction: {stats['size_reduction']:.1f}%")
    
    # Print timing statistics for all centers
    print(f"\nTiming Performance ({len(centers_timing)} center positions):")
    
    avg_time = np.mean(compression_times)
    std_time = np.std(compression_times)
    print(f"Average compression time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Min/Max time: {min(compression_times):.2f} / {max(compression_times):.2f} ms")
    
    # Calculate throughput metrics
    fps = 1000 / avg_time  # Frames per second
    megapixels_per_sec = (stats['input_pixels'] / 1e6) * fps
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Processing rate: {megapixels_per_sec:.1f} megapixels/second")

if __name__ == "__main__":
    print("Testing foveated downsampling functionality...")
    test_foveated_downsampling()