#!/usr/bin/env python3
"""
Quantitative Edge Detection Analysis
Provides statistical analysis of edge detection performance across datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

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

def analyze_dataset_edge_strength(dataset_path, dataset_type):
    """
    Analyze edge strength statistics for a dataset.
    """
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
    
    images_dir = os.path.join(dataset_path, "images")
    
    if not os.path.exists(images_dir):
        print(f"  ⚠️ Dataset not found: {dataset_path}")
        return None
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    
    if len(image_files) == 0:
        print(f"  ⚠️ No images found in {dataset_path}")
        return None
    
    # Collect edge statistics
    h_edge_means = []
    v_edge_means = []
    combined_edge_means = []
    max_edge_strengths = []
    
    print(f"  Analyzing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc=f"  {dataset_type}", leave=False):
        img_path = os.path.join(images_dir, img_file)
        img = np.array(Image.open(img_path))
        
        # Convert to grayscale
        gray = np.mean(img, axis=2)
        
        # Apply Sobel kernels
        h_edges = conv2d(gray, sobel_x)
        v_edges = conv2d(gray, sobel_y)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Collect statistics
        h_edge_means.append(np.mean(np.abs(h_edges)))
        v_edge_means.append(np.mean(np.abs(v_edges)))
        combined_edge_means.append(np.mean(edges))
        max_edge_strengths.append(np.max(edges))
    
    return {
        'h_edge_means': h_edge_means,
        'v_edge_means': v_edge_means,
        'combined_edge_means': combined_edge_means,
        'max_edge_strengths': max_edge_strengths,
        'dataset_type': dataset_type
    }

def generate_quantitative_analysis():
    """
    Generate quantitative analysis of edge detection across all datasets.
    """
    print("📊 Quantitative Edge Detection Analysis")
    print("=" * 60)
    
    # Define datasets to analyze
    datasets = [
        ('datasets/rectangles_grayscale', 'grayscale'),
        ('datasets/rectangles_colored', 'colored'),
        ('datasets/rectangles_grayscale_border', 'grayscale_border'),
        ('datasets/rectangles_colored_border', 'colored_border')
    ]
    
    results = []
    
    for dataset_path, dataset_type in datasets:
        print(f"\nAnalyzing {dataset_type}...")
        result = analyze_dataset_edge_strength(dataset_path, dataset_type)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("\n❌ No datasets found. Generate datasets first:")
        print("   ./run generate_all")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantitative Edge Detection Analysis Across Datasets', 
                 fontsize=14, fontweight='bold')
    
    dataset_types = [r['dataset_type'] for r in results]
    
    # Plot 1: Combined edge strength distribution
    ax1 = axes[0, 0]
    for result in results:
        ax1.hist(result['combined_edge_means'], alpha=0.6, label=result['dataset_type'], bins=20)
    ax1.set_xlabel('Mean Edge Strength')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Combined Edge Strength Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of edge strengths
    ax2 = axes[0, 1]
    combined_data = [r['combined_edge_means'] for r in results]
    bp = ax2.boxplot(combined_data, labels=dataset_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Mean Edge Strength')
    ax2.set_title('Edge Strength by Dataset Type')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 3: Horizontal vs Vertical edge strength
    ax3 = axes[1, 0]
    for result in results:
        ax3.scatter(result['h_edge_means'], result['v_edge_means'], 
                   alpha=0.6, label=result['dataset_type'], s=50)
    ax3.plot([0, max([max(r['h_edge_means']) for r in results])],
             [0, max([max(r['v_edge_means']) for r in results])],
             'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('Horizontal Edge Strength')
    ax3.set_ylabel('Vertical Edge Strength')
    ax3.set_title('Horizontal vs Vertical Edge Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Max edge strength comparison
    ax4 = axes[1, 1]
    max_means = [np.mean(r['max_edge_strengths']) for r in results]
    max_stds = [np.std(r['max_edge_strengths']) for r in results]
    bars = ax4.bar(dataset_types, max_means, yerr=max_stds, 
                   capsize=5, alpha=0.7, color=['blue', 'red', 'green', 'orange'])
    ax4.set_ylabel('Max Edge Strength')
    ax4.set_title('Maximum Edge Strength by Dataset')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, mean in zip(bars, max_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('quantitative_edge_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis complete! Saved to: quantitative_edge_analysis.png")
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 60)
    for result in results:
        dt = result['dataset_type']
        print(f"\n{dt}:")
        print(f"  Mean combined edge: {np.mean(result['combined_edge_means']):.2f} ± {np.std(result['combined_edge_means']):.2f}")
        print(f"  Mean horizontal edge: {np.mean(result['h_edge_means']):.2f} ± {np.std(result['h_edge_means']):.2f}")
        print(f"  Mean vertical edge: {np.mean(result['v_edge_means']):.2f} ± {np.std(result['v_edge_means']):.2f}")
        print(f"  Mean max edge: {np.mean(result['max_edge_strengths']):.2f} ± {np.std(result['max_edge_strengths']):.2f}")
    
    print("\n📊 Generated files:")
    print("   • quantitative_edge_analysis.png - Statistical comparison")
    print("   • edge_detection_analysis.png (from kernel_analysis)")
    print("   • edge_profile_comparison.png (from kernel_analysis)")

if __name__ == "__main__":
    generate_quantitative_analysis()
