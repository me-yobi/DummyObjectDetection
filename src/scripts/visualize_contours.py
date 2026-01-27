#!/usr/bin/env python3
"""Visualize contours and intermediate edge maps produced by the detector"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ..models.detector import SimpleRectangleDetector

def load_image(image_path):
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def visualize_pipeline(image_path, output_path=None):
    """Visualize the full pipeline: edges, threshold, dilation, contours"""
    # Load image
    img = load_image(image_path)
    detector = SimpleRectangleDetector()
    
    # --- Step 1: Convert to grayscale ---
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.copy()
    
    # --- Step 2: Compute Sobel edges ---
    h_edges = detector.conv2d(gray, detector.h_kernel)
    v_edges = detector.conv2d(gray, detector.v_kernel)
    edges = np.sqrt(h_edges**2 + v_edges**2)
    
    # --- Step 3: Adaptive threshold ---
    threshold = np.mean(edges) + 2 * np.std(edges)
    edge_map = (edges > threshold).astype(np.uint8) * 255
    
    # --- Step 4: Morphological dilation ---
    morph_kernel = np.ones((3, 3), dtype=np.float32)
    dilated = detector.conv2d(edge_map.astype(np.float32), morph_kernel)
    dilated = (dilated > 0).astype(np.uint8) * 255
    
    # --- Step 5: Find contours ---
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours (same logic as detector)
    h_img, w_img = img.shape[:2]
    image_area = h_img * w_img
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area >= 0.5 * image_area:
            continue
        valid_contours.append(cnt)
    
    # Select largest valid contour (if any)
    if valid_contours:
        best_contour = max(valid_contours, key=cv2.contourArea)
        # Get bounding box
        x, y, w, h = cv2.boundingRect(best_contour)
        pred_box = [1.0, (x + w/2)/w_img, (y + h/2)/h_img, w/w_img, h/h_img]
    else:
        best_contour = None
        pred_box = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Contour Pipeline Visualization\nImage: {image_path}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Sobel edges
    im1 = axes[0, 1].imshow(edges, cmap='viridis')
    axes[0, 1].set_title('Sobel Edge Magnitude')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Binary edge map after threshold
    axes[0, 2].imshow(edge_map, cmap='gray')
    axes[0, 2].set_title(f'Binary Edges (threshold={threshold:.2f})')
    axes[0, 2].axis('off')
    
    # Dilated edges
    axes[1, 0].imshow(dilated, cmap='gray')
    axes[1, 0].set_title('Dilated Edges (3x3 kernel)')
    axes[1, 0].axis('off')
    
    # All contours on original
    img_all_contours = img.copy()
    cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 1)  # Blue for all
    axes[1, 1].imshow(img_all_contours)
    axes[1, 1].set_title(f'All Contours ({len(contours)} found)')
    axes[1, 1].axis('off')
    
    # Final selected contour + bounding box
    img_final = img.copy()
    if best_contour is not None:
        cv2.drawContours(img_final, [best_contour], -1, (0, 255, 0), 2)  # Green for selected
        # Draw bounding box
        x1 = int((pred_box[1] - pred_box[3]/2) * w_img)
        y1 = int((pred_box[2] - pred_box[4]/2) * h_img)
        x2 = int((pred_box[1] + pred_box[3]/2) * w_img)
        y2 = int((pred_box[2] + pred_box[4]/2) * h_img)
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1, 2].set_title(f'Selected Contour (IoU-based)\nBox: [{pred_box[1]:.3f}, {pred_box[2]:.3f}, {pred_box[3]:.3f}, {pred_box[4]:.3f}]')
    else:
        axes[1, 2].set_title('No Valid Contour Found')
    axes[1, 2].imshow(img_final)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize contours and edge maps')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save visualization (optional)')
    args = parser.parse_args()
    
    visualize_pipeline(args.image, args.output)

if __name__ == "__main__":
    main()
