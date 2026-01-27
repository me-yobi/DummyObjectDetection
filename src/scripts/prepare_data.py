#!/usr/bin/env python3
"""
Script to prepare the dataset using DataGenerator before training
"""
import sys
import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

try:
    from ..data.DataGenerator import generate_rectangle_dataset
except ImportError:
    print("Error: Could not import DataGenerator from src.data")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--output-dir', type=str, default='datasets/rectangles', 
                       help='Output directory for generated dataset')
    parser.add_argument('--num-images', type=int, default=500, 
                       help='Number of images to generate')
    parser.add_argument('--force', action='store_true', 
                       help='Force regeneration even if dataset exists')
    
    args = parser.parse_args()
    
    # Check if dataset already exists
    if os.path.exists(args.output_dir) and not args.force:
        images_dir = os.path.join(args.output_dir, 'images')
        labels_dir = os.path.join(args.output_dir, 'labels')
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            num_existing = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            if num_existing >= args.num_images:
                print(f"Dataset already exists with {num_existing} images")
                print(f"Use --force to regenerate or --num-images to request more")
                return
    
    print(f"Generating {args.num_images} images in {args.output_dir}...")
    
    # Generate the dataset
    generate_rectangle_dataset(
        output_folder=args.output_dir,
        num_images=args.num_images
    )
    
    print(f"\nDataset preparation complete!")
    print(f"Images: {len(os.listdir(os.path.join(args.output_dir, 'images')))}")
    print(f"Labels: {len(os.listdir(os.path.join(args.output_dir, 'labels')))}")

if __name__ == "__main__":
    main()
