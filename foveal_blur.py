import cv2
import numpy as np

def foveal_blur(image, fovea_center, output_size=(64, 64), crop_size=(180, 180), fovea_radius_norm=0.2, max_blur=10):
    """
    Applies a fast, correct foveal blur using a pyramid blending technique.
    
    Args:
        image (np.ndarray): 272x272 input image (BGR)
        fovea_center (tuple): (x, y) coordinates of fixation point
        output_size (tuple): Dimensions of the output image (w, h)
        crop_size (tuple): The size of the area to crop from the source image before resizing.
        fovea_radius_norm (float): Radius of the sharp foveal region, normalized to [0, 1]
        max_blur (int): Maximum blur sigma at the periphery
    
    Returns:
        np.ndarray: Foveated and downscaled output image
    """
    h, w = output_size[1], output_size[0]
    crop_h, crop_w = crop_size[1], crop_size[0]
    cx, cy = fovea_center
    
    # 1. Crop a larger area around the fovea center
    start_x = int(cx - crop_w//2)
    start_y = int(cy - crop_h//2)
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # Create intermediate canvas with black padding
    cropped_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    
    # Handle boundary conditions for cropping
    src_x1, src_y1 = max(0, start_x), max(0, start_y)
    src_x2, src_y2 = min(image.shape[1], end_x), min(image.shape[0], end_y)
    dst_x1, dst_y1 = max(0, -start_x), max(0, -start_y)
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        cropped_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # 2. Resize the cropped area to the final output size
    output = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_AREA)

    # 3. Create a pyramid of blurred images from the resized output
    num_levels = 8
    pyramid = [output]
    for i in range(1, num_levels):
        sigma = (i / (num_levels - 1)) * max_blur
        ksize = int(2 * np.ceil(3 * sigma) + 1) # Ensure kernel is large enough
        blurred_level = cv2.GaussianBlur(output, (ksize, ksize), sigma)
        pyramid.append(blurred_level)
    pyramid = np.stack(pyramid).astype(np.float32)

    # 4. Create a blend map based on distance from the center
    center_x, center_y = w//2, h//2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    norm_distances = distances / max_distance
    
    # Create a sharp foveal region
    blur_strength = (norm_distances - fovea_radius_norm) / (1.0 - fovea_radius_norm)
    blur_strength = np.clip(blur_strength, 0, 1)
    
    # Map blur strength to pyramid indices
    pyramid_idx_float = blur_strength * (num_levels - 1)
    
    # 5. Vectorized blending of pyramid levels
    low_idx = np.floor(pyramid_idx_float).astype(int)
    high_idx = np.ceil(pyramid_idx_float).astype(int)
    alpha = (pyramid_idx_float - low_idx)[..., np.newaxis]

    # Reshape for vectorized gathering
    h_w = h * w
    pyramid_flat = pyramid.reshape(num_levels, h_w, 3)
    low_idx_flat = low_idx.flatten()
    high_idx_flat = high_idx.flatten()
    pixel_indices = np.arange(h_w)

    # Gather pixels from low and high blur levels
    low_pixels = pyramid_flat[low_idx_flat, pixel_indices]
    high_pixels = pyramid_flat[high_idx_flat, pixel_indices]
    
    # Interpolate and reshape back to image dimensions
    alpha_flat = alpha.reshape(h_w, 1)
    blended_pixels = (1.0 - alpha_flat) * low_pixels + alpha_flat * high_pixels
    result = blended_pixels.reshape(h, w, 3)

    return np.clip(result, 0, 255).astype(np.uint8)
