import os
import cv2
import numpy as np
from tqdm import tqdm

def create_noisy_image_with_rectangle(image_size=(256, 256, 3), min_rect_size=32, max_rect_size=128):
    """
    Creates a noisy image with a randomly placed and colored rectangle.
    
    Args:
        image_size: Tuple of (height, width, channels) for the output image
        min_rect_size: Minimum size of the rectangle
        max_rect_size: Maximum size of the rectangle
        
    Returns:
        A tuple of (image, label) where label is in YOLO format
    """
    # Create a blank image with gray background (less noise for better visibility)
    background = np.random.randint(100, 156, size=image_size, dtype=np.uint8)  # Gray background
    
    # Generate random rectangle dimensions and position
    rect_width = np.random.randint(min_rect_size, max_rect_size)
    rect_height = np.random.randint(min_rect_size, max_rect_size)
    
    # Ensure the rectangle stays within image bounds
    x1 = np.random.randint(0, image_size[1] - rect_width)
    y1 = np.random.randint(0, image_size[0] - rect_height)
    x2 = x1 + rect_width
    y2 = y1 + rect_height
    
    # Choose a high-contrast color for the rectangle (not gray)
    # Use either bright or dark colors to ensure visibility
    bright_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], 
                    [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    dark_colors = [[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]]
    
    if np.random.random() > 0.5:
        # Bright color
        border_color = tuple(bright_colors[np.random.randint(len(bright_colors))])
    else:
        # Dark color
        border_color = tuple(dark_colors[np.random.randint(len(dark_colors))])
    
    border_thickness = 3  # Fixed thickness for consistency
    
    # Draw the rectangle on the background
    cv2.rectangle(background, (x1, y1), (x2, y2), border_color, border_thickness)
    
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

def generate_rectangle_dataset(output_folder="datasets/rectangles", num_images=500):
    """
    Generates a dataset of noisy images with random rectangles.
    
    Args:
        output_folder: Where to save the generated dataset
        num_images: Number of images to generate
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
        img, label = create_noisy_image_with_rectangle()
        
        # Save the image
        img_filename = f"rectangle_{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, img)
        
        # Save the label in YOLO format
        label_filename = f"rectangle_{i:04d}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")
    
    print(f"\nSuccessfully generated {num_images} images in {output_folder}")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")

if __name__ == "__main__":
    # Generate the dataset
    generate_rectangle_dataset()