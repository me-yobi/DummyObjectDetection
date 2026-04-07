#!/usr/bin/env python3
"""
Edge Detection Kernel Analysis
Analyzes Sobel edge detection behavior on different rectangle types
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_test_rectangle(image_size=256, rect_size=100, dataset_type='grayscale'):
    """
    Create a test image with a rectangle for kernel analysis.
    """
    if dataset_type == 'grayscale':
        # Grayscale noise background
        background = np.random.randint(100, 156, size=(image_size, image_size, 3), dtype=np.uint8)
        background = np.stack([background[:,:,0]] * 3, axis=-1)
        border_color = (0, 0, 0)
    elif dataset_type == 'grayscale_border':
        # Grayscale noise background
        background = np.random.randint(100, 156, size=(image_size, image_size, 3), dtype=np.uint8)
        background = np.stack([background[:,:,0]] * 3, axis=-1)
        border_color = (0, 0, 0)
    else:  # colored
        # Colored noise background
        background = np.random.randint(100, 156, size=(image_size, image_size, 3), dtype=np.uint8)
        border_color = (255, 0, 0)  # Red for visibility
    
    # Place rectangle in center
    start = (image_size - rect_size) // 2
    end = start + rect_size
    
    if 'border' in dataset_type:
        # Draw border only
        thickness = 3
        background[start:end, start:end] = border_color
        for i in range(thickness):
            if start-i >= 0: background[start-i, start:end] = border_color
            if end+i < image_size: background[end+i, start:end] = border_color
            if start-i >= 0: background[start:end, start-i] = border_color
            if end+i < image_size: background[start:end, end+i] = border_color
    else:
        # Draw filled rectangle
        background[start:end, start:end] = border_color
    
    return background.astype(np.uint8)

def conv2d(image, kernel):
    """2D convolution with valid padding"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    out_h = h - kh + 1
    out_w = w - kw + 1
    
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

def analyze_edge_detection():
    """
    Analyze Sobel edge detection behavior on different rectangle types.
    """
    print("🔬 Analyzing Edge Detection Kernel Behavior")
    print("=" * 60)
    
    # Define Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # Test on different rectangle types
    dataset_types = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Edge Detection Kernel Analysis', fontsize=16, fontweight='bold')
    
    for idx, dataset_type in enumerate(dataset_types):
        print(f"\nAnalyzing {dataset_type}...")
        
        # Create test image
        img = create_test_rectangle(dataset_type=dataset_type)
        
        # Convert to grayscale
        gray = np.mean(img, axis=2)
        
        # Apply Sobel kernels
        h_edges = conv2d(gray, sobel_x)
        v_edges = conv2d(gray, sobel_y)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Plot results
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'{dataset_type}\nOriginal')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(h_edges, cmap='RdBu')
        axes[idx, 1].set_title('Sobel X (Horizontal Edges)')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(v_edges, cmap='RdBu')
        axes[idx, 2].set_title('Sobel Y (Vertical Edges)')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(edges, cmap='hot')
        axes[idx, 3].set_title('Combined Edge Strength')
        axes[idx, 3].axis('off')
        
        # Print statistics
        print(f"  Horizontal edge mean: {np.mean(np.abs(h_edges)):.2f}")
        print(f"  Vertical edge mean: {np.mean(np.abs(v_edges)):.2f}")
        print(f"  Combined edge mean: {np.mean(edges):.2f}")
        print(f"  Max edge strength: {np.max(edges):.2f}")
    
    plt.tight_layout()
    plt.savefig('edge_detection_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis complete! Saved to: edge_detection_analysis.png")
    
    # Create edge profile comparison
    create_edge_profile_comparison(sobel_x, sobel_y)
    
    print("\n📊 Generated files:")
    print("   • edge_detection_analysis.png - Full kernel visualization")
    print("   • edge_profile_comparison.png - Edge strength profiles")

def create_edge_profile_comparison(sobel_x, sobel_y):
    """
    Create edge strength profile comparison across dataset types.
    """
    dataset_types = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Edge Strength Profile Comparison', fontsize=14, fontweight='bold')
    
    for idx, dataset_type in enumerate(dataset_types):
        row = idx // 2
        col = idx % 2
        
        img = create_test_rectangle(dataset_type=dataset_type)
        gray = np.mean(img, axis=2)
        
        h_edges = conv2d(gray, sobel_x)
        v_edges = conv2d(gray, sobel_y)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Extract center line profile
        center_row = edges.shape[0] // 2
        profile = edges[center_row, :]
        
        axes[row, col].plot(profile, linewidth=2)
        axes[row, col].set_title(f'{dataset_type} - Center Edge Profile')
        axes[row, col].set_xlabel('Pixel Position')
        axes[row, col].set_ylabel('Edge Strength')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics
        mean_strength = np.mean(profile)
        max_strength = np.max(profile)
        axes[row, col].text(0.02, 0.98, f'Mean: {mean_strength:.2f}\nMax: {max_strength:.2f}',
                          transform=axes[row, col].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('edge_profile_comparison.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    analyze_edge_detection()
