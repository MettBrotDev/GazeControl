import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from foveal_blur import foveal_blur
import cv2

def create_test_image(size=(272, 272)):
    """Create a test image with patterns (matching DownsamplingBlur.py)."""
    height, width = size
    
    # Create a numpy array with same patterns as DownsamplingBlur
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some circles and patterns
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Checkerboard pattern
    checker = ((x // 16) + (y // 16)) % 2
    img[:, :, 0] = checker * 255 * 0.8  # Red channel
    
    # Circular pattern
    center = height // 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    circle = (np.sin(radius / 10) + 1) / 2
    img[:, :, 1] = circle * 255 * 0.7  # Green channel
    
    # Diagonal stripes
    diagonal = np.sin((x + y) / 8)
    img[:, :, 2] = ((diagonal + 1) / 2) * 255 * 0.6  # Blue channel
    
    return img.astype(np.uint8)

def test_foveal_blur_cv():
    """Test the OpenCV-based foveal blur implementation."""
    # Create test image
    test_img = create_test_image((272, 272))
    
    # Test different center positions (same as DownsamplingBlur.py)
    centers_visual = [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]
    centers_timing = [(0.1, 0.1), (0.2, 0.3), (0.3, 0.7), (0.4, 0.2), (0.5, 0.5), 
                     (0.6, 0.8), (0.7, 0.4), (0.8, 0.7), (0.9, 0.6), (0.5, 0.1),
                     (0.1, 0.5), (0.9, 0.9), (0.3, 0.3), (0.7, 0.7), (0.2, 0.8)]
    
    output_size = (64, 64)
    crop_size = (128, 128)  # Capture a larger area from the source image
    fovea_radius_norm = 0.25
    max_blur = 8
    
    # Storage for timing measurements
    compression_times = []
    
    # Set matplotlib backend for non-interactive environments
    import matplotlib
    matplotlib.use('Agg')
    
    # First run timing tests on all centers
    print("Running timing tests on multiple center positions...")
    for center_x, center_y in centers_timing:
        # Convert to pixel coordinates
        fx = center_x * 272
        fy = center_y * 272
        
        start_time = time.time()
        
        blurred = foveal_blur(test_img, (fx, fy), output_size, crop_size, fovea_radius_norm, max_blur)
        
        end_time = time.time()
        compression_time = (end_time - start_time) * 1000  # Convert to milliseconds
        compression_times.append(compression_time)
    
    # Then create visualizations for just the visual centers
    fig, axes = plt.subplots(len(centers_visual), 4, figsize=(16, 4*len(centers_visual)))
    if len(centers_visual) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (center_x, center_y) in enumerate(centers_visual):
        # Convert to pixel coordinates
        fx = center_x * 272
        fy = center_y * 272
        
        # Apply blur
        blurred = foveal_blur(test_img, (fx, fy), output_size, crop_size, fovea_radius_norm, max_blur)
        
        # Create blur strength visualization for display
        h_out, w_out = output_size[1], output_size[0]
        center_y_out, center_x_out = h_out // 2, w_out // 2
        y, x = np.ogrid[:h_out, :w_out]
        distances = np.sqrt((x - center_x_out)**2 + (y - center_y_out)**2)
        max_distance = np.sqrt(center_x_out**2 + center_y_out**2)
        norm_distances = distances / max_distance
        blur_strength = (norm_distances - fovea_radius_norm) / (1.0 - fovea_radius_norm)
        sigma_map = np.clip(blur_strength, 0, 1) * max_blur
        
        # Create distance map from original fovea position for comparison
        y_orig, x_orig = np.indices((272, 272))
        distance_orig = np.sqrt((x_orig - fx)**2 + (y_orig - fy)**2)
        
        # Plot results
        # Original image
        axes[i, 0].imshow(test_img)
        axes[i, 0].set_title(f'Original Image (272x272)')
        axes[i, 0].axis('off')
        
        # Add center point
        center_px_x = int(center_x * 272)
        center_px_y = int(center_y * 272)
        axes[i, 0].plot(center_px_x, center_px_y, 'r+', markersize=20, markeredgewidth=3)
        
        # Foveal blur result
        axes[i, 1].imshow(blurred, interpolation='none')
        axes[i, 1].set_title(f'Foveal Blur ({w_out}x{h_out})\nCenter: ({center_x:.1f}, {center_y:.1f})')
        axes[i, 1].axis('off')
        # Mark fovea center (should be at center of output image)
        axes[i, 1].plot(center_x_out, center_y_out, 'r+', markersize=10, markeredgewidth=2)
        
        # Blur strength map (from output image perspective)
        im2 = axes[i, 2].imshow(sigma_map, cmap='viridis', interpolation='none', vmin=0, vmax=max_blur)
        axes[i, 2].set_title(f'Blur Strength Map ({w_out}x{h_out})')
        axes[i, 2].axis('off')
        axes[i, 2].plot(center_x_out, center_y_out, 'r+', markersize=10, markeredgewidth=2)
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Distance map from original fovea position
        im3 = axes[i, 3].imshow(distance_orig, cmap='plasma', interpolation='none')
        axes[i, 3].set_title(f'Distance from Fovea (272x272)')
        axes[i, 3].axis('off')
        axes[i, 3].plot(fx, fy, 'r+', markersize=10, markeredgewidth=2)
        plt.colorbar(im3, ax=axes[i, 3])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('./foveal_blur_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'foveal_blur_test.png'")
    plt.close()
    
    # Print compression statistics
    input_pixels = test_img.shape[0] * test_img.shape[1]
    output_pixels = blurred.shape[0] * blurred.shape[1]
    spatial_compression = input_pixels / output_pixels
    size_reduction = (1 - output_pixels / input_pixels) * 100
    
    print("\nFoveal Blur Test Results:")
    print(f"Input image size: {test_img.shape}")
    print(f"Output image shape: {blurred.shape}")
    print(f"  - RGB channels: {blurred.shape[2]}")
    
    print(f"\nCompression Statistics:")
    print(f"Input pixels: {input_pixels:,}")
    print(f"Output pixels: {output_pixels:,}")
    print(f"Spatial compression ratio: {spatial_compression:.1f}x")
    print(f"Size reduction: {size_reduction:.1f}%")
    
    # Print timing statistics for all centers
    print(f"\nTiming Performance ({len(centers_timing)} center positions):")
    
    avg_time = np.mean(compression_times)
    std_time = np.std(compression_times)
    print(f"Average blur time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Min/Max time: {min(compression_times):.2f} / {max(compression_times):.2f} ms")
    
    # Calculate throughput metrics
    fps = 1000 / avg_time  # Frames per second
    megapixels_per_sec = (input_pixels / 1e6) * fps
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Processing rate: {megapixels_per_sec:.1f} megapixels/second")
    
    # Print blur function info
    print(f"\nBlur Function Info:")
    print(f"Crop size from source: {crop_size}")
    print(f"Output size: {output_size}")
    print(f"Fovea radius (normalized): {fovea_radius_norm}")
    print(f"Max blur sigma: {max_blur}")
    print("Note: Output image is centered around the fovea position!")

if __name__ == "__main__":
    print("Testing foveal blur functionality...")
    test_foveal_blur_cv()

