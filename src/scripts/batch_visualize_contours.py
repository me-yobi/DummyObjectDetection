#!/usr/bin/env python3
"""Batch process random images from the dataset with contour visualization"""

import os
import random
import argparse
from pathlib import Path
from .visualize_contours import visualize_pipeline

def batch_visualize_contours(num_samples=5, dataset_dir="datasets/rectangles", output_dir="contour_visualizations"):
    """Run contour visualization on random images from dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_dir = os.path.join(dataset_dir, "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Select random samples
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Processing {len(selected_files)} random images...")
    
    for i, filename in enumerate(selected_files, 1):
        print(f"[{i}/{len(selected_files)}] Processing {filename}")
        
        # Input and output paths
        input_path = os.path.join(image_dir, filename)
        output_filename = f"contour_{filename.replace('.jpg', '.png').replace('.png', '.png')}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Run visualization
            visualize_pipeline(input_path, output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
    
    print(f"\nBatch processing complete! Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch visualize contours on random dataset images')
    parser.add_argument('--num', type=int, default=5, help='Number of random images to process')
    parser.add_argument('--dataset', type=str, default='datasets/rectangles', help='Dataset directory path')
    parser.add_argument('--output', type=str, default='contour_visualizations', help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    batch_visualize_contours(args.num, args.dataset, args.output)

if __name__ == "__main__":
    main()
