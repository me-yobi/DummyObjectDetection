import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_noisy_image_with_rectangle(image_size=(256, 256, 3), min_rect_size=32, max_rect_size=128, dataset_type='colored'):
    """
    Creates a noisy image with a randomly placed rectangle.
    
    Args:
        image_size: Tuple of (height, width, channels) for the output image
        min_rect_size: Minimum size of the rectangle
        max_rect_size: Maximum size of the rectangle
        dataset_type: 'grayscale' for grayscale noise with black rectangle, 'colored' for colored noise with colored rectangle
        
    Returns:
        A tuple of (image, label) where label is in YOLO format
    """
    if dataset_type == 'grayscale':
        # Grayscale noise background
        background = np.random.randint(50, 200, size=image_size, dtype=np.uint8)
        # Convert to grayscale by copying the first channel to all channels
        background = np.stack([background[:,:,0]] * 3, axis=-1)
        # Black rectangle
        border_color = (0, 0, 0)
    else:  # colored
        # Colored noise background
        background = np.random.randint(100, 156, size=image_size, dtype=np.uint8)
        # Choose a high-contrast color for the rectangle (not gray)
        bright_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], 
                        [255, 255, 0], [255, 0, 255], [0, 255, 255]]
        dark_colors = [[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]]
        
        if np.random.random() > 0.5:
            border_color = tuple(bright_colors[np.random.randint(len(bright_colors))])
        else:
            border_color = tuple(dark_colors[np.random.randint(len(dark_colors))])
    
    # Generate random rectangle dimensions and position
    rect_width = np.random.randint(min_rect_size, max_rect_size)
    rect_height = np.random.randint(min_rect_size, max_rect_size)
    
    # Ensure the rectangle stays within image bounds
    x1 = np.random.randint(0, image_size[1] - rect_width)
    y1 = np.random.randint(0, image_size[0] - rect_height)
    x2 = x1 + rect_width
    y2 = y1 + rect_height
    
    
    border_thickness = 3  # Fixed thickness for consistency
    
    # Draw rectangle on background using numpy
    background[y1:y2+1, x1:x2+1] = border_color
    # Make the border thicker by expanding outward
    for i in range(border_thickness):
        if y1-i >= 0: background[y1-i, x1:x2+1] = border_color
        if y2+i < image_size[0]: background[y2+i, x1:x2+1] = border_color
        if x1-i >= 0: background[y1:y2+1, x1-i] = border_color
        if x2+i < image_size[1]: background[y1:y2+1, x2+i] = border_color
    
    # Add some noise but keep the rectangle visible
    noise = np.random.randint(-30, 31, size=image_size, dtype=np.int16)
    noisy_image = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Calculate YOLO format label: [class, x_center, y_center, width, height] (all normalized)
    image_height, image_width = image_size[:2]
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = rect_width / image_width
    height = rect_height / image_height
    
    # Using class_id = 0 for rectangle
    label = [0, x_center, y_center, width, height]
    
    return noisy_image, label

def generate_rectangle_dataset(output_folder="datasets/rectangles", num_images=500, dataset_type='colored'):
    """
    Generates a dataset of noisy images with random rectangles.
    
    Args:
        output_folder: Where to save the generated dataset
        num_images: Number of images to generate
        dataset_type: 'grayscale' for grayscale noise with black rectangle, 'colored' for colored noise with colored rectangle
    """
    # Set up directory structure
    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")
    
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Creating {num_images} images in {output_folder}...")
    
    for i in tqdm(range(num_images), desc="Generating images"):
        # Create the image and get its label
        img, label = create_noisy_image_with_rectangle(dataset_type=dataset_type)
        
        # Save the image
        img_filename = f"rectangle_{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        Image.fromarray(img.astype(np.uint8)).save(img_path)
        
        # Save the label in YOLO format
        label_filename = f"rectangle_{i:04d}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")
    
    print(f"\nSuccessfully generated {num_images} images in {output_folder}")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")

if __name__ == "__main__":
    # Generate both datasets
    print("Generating grayscale dataset...")
    generate_rectangle_dataset(output_folder="datasets/rectangles_grayscale", num_images=500, dataset_type='grayscale')
    print("\nGenerating colored dataset...")
    generate_rectangle_dataset(output_folder="datasets/rectangles_colored", num_images=500, dataset_type='colored')