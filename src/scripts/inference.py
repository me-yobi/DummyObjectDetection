#!/usr/bin/env python3
"""Run inference on a single image using the trained model"""

import argparse
import cv2
import numpy as np
from .config import Config
from ..models.detector import SimpleRectangleDetector
from ..utils.visualization import plot_image_with_boxes

def load_image(image_path):
    """Load and preprocess image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def run_inference(model, image):
    """Run model inference on image"""
    # Ensure image is in HWC format
    if len(image.shape) == 3 and image.shape[0] == 3:
        # CHW format, convert to HWC
        image = image.transpose(1, 2, 0)
    
    # Forward pass
    model.eval()
    output = model.forward(image)
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='saved_models/best_model.npy', 
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default='inference_result.jpg',
                       help='Path to save output image')
    parser.add_argument('--config', type=str, default='config.py', help='Config file')
    
    args = parser.parse_args()
    
    # Load model (analytic detector - no trainable weights)
    model = SimpleRectangleDetector()
    
    # Note: SimpleRectangleDetector has no learned weights; args.model is ignored.
    if os.path.exists(args.model):
        print(f"Ignoring saved model file {args.model}: SimpleRectangleDetector has no weights to load.")
    
    # Load image
    try:
        image = load_image(args.image)
        print(f"Loaded image: {args.image}")
    except ValueError as e:
        print(e)
        return
    
    # Run inference
    output = run_inference(model, image)
    print(f"Model output: {output}")
    
    # Denormalize output for visualization
    pred_box = output[1:] * Config.IMAGE_SIZE
    
    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    x1 = pred_box[0] - pred_box[2] / 2
    y1 = pred_box[1] - pred_box[3] / 2
    x2 = pred_box[0] + pred_box[2] / 2
    y2 = pred_box[1] + pred_box[3] / 2
    
    # Denormalize image for visualization
    img_vis = (image * 255).astype(np.uint8)
    
    # Draw predicted box
    cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    # Add label
    label = f"Object: {output[0]:.2f}"
    cv2.putText(img_vis, label, (int(x1), int(y1)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save output
    # Convert RGB to BGR for OpenCV
    img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, img_vis_bgr)
    print(f"Saved result to: {args.output}")
    
    # Also display using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img_vis)
        plt.title(f'Prediction: {output}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping visualization")

if __name__ == "__main__":
    main()
