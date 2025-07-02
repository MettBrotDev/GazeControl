import os
import random
from PIL import Image
from pathlib import Path

def process_images(base_dir, output_base_dir="./Data/labeled"):
    os.makedirs(output_base_dir, exist_ok=True)
    task_name = Path(base_dir).name
    task_name = task_name.replace("task_", "")  # Remove 'task_' prefix for labeling
    
    # all directories given by the dataset generation script
    subdirs = ["test", "test_gen", "train", "val"]
    
    for subdir in subdirs:
        src_dir = os.path.join(base_dir, subdir)
        dest_dir = os.path.join(output_base_dir, subdir)
        
        if not os.path.exists(src_dir):
            print(f"Directory {src_dir} does not exist, skipping.")
            continue
        
        os.makedirs(dest_dir, exist_ok=True)
        
        # Process each image in the current subdirectory
        for filename in os.listdir(src_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_image(os.path.join(src_dir, filename), dest_dir, task_name)

def process_image(image_path, output_dir, task_name):
    img = Image.open(image_path)
    width, height = img.size
    
    part_width = width // 4
    
    # Extract the 4 parts
    parts = []
    for i in range(4):
        box = (i * part_width, 0, (i + 1) * part_width, height)
        parts.append(img.crop(box))
    
    # Create a list of indices and shuffle
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    
    # The new position of the odd part (which the dataset always puts in the last position)
    new_pos = indices.index(3)
    
    # Create a new image with 2x2 dimensions
    new_width = part_width * 2
    new_height = height * 2
    new_img = Image.new(img.mode, (new_width, new_height))
    
    for i, idx in enumerate(indices):
        # 0, 1
        # 2, 3
        row = i // 2  
        col = i % 2   
        
        x_position = col * part_width
        y_position = row * height
        
        new_img.paste(parts[idx], (x_position, y_position))
    
    # label the new image
    original_name = Path(image_path).stem
    new_filename = f"{task_name}_{original_name}_{new_pos}.png"
    
    new_img.save(os.path.join(output_dir, new_filename))

if __name__ == "__main__":
    data_dir = "./Data"
    for entry in os.listdir(data_dir):
        task_dir = os.path.join(data_dir, entry)
        if os.path.isdir(task_dir) and entry.startswith("task_"):
            print(f"Processing task directory: {task_dir}")
            process_images(task_dir)
