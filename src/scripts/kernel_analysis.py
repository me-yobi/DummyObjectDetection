#!/usr/bin/env python3
"""
Kernel Behavior Analysis: Why Border Datasets Outperform Filled Datasets

This script analyzes how the Sobel edge detection kernels interact differently
with filled vs border-only rectangles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.DataGenerator import create_noisy_image_with_rectangle
from src.models.detector import SimpleRectangleDetector


def analyze_kernel_response(dataset_type, num_samples=10):
    """
    Analyze how kernels respond to different dataset types.
    Returns metrics about edge strength, contour quality, and detection accuracy.
    """
    detector = SimpleRectangleDetector()
    
    metrics = {
        'edge_strength_mean': [],
        'edge_strength_std': [],
        'corner_response_mean': [],
        'corner_response_max': [],
        'contour_count': [],
        'contour_area_ratio': [],
        'threshold_ratio': [],
        'detection_success': [],
        'iou_scores': []
    }
    
    for i in range(num_samples):
        # Generate image and label
        image, true_label = create_noisy_image_with_rectangle(dataset_type=dataset_type)
        
        # Convert to grayscale for processing
        gray = np.mean(image, axis=2)
        
        # Apply Sobel kernels
        h_edges = detector.conv2d(gray, detector.h_kernel)
        v_edges = detector.conv2d(gray, detector.v_kernel)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Apply Laplacian corner detector
        corners = detector.conv2d(gray, detector.c_kernel)
        corners = np.abs(corners)  # Laplacian gives signed response, take absolute
        
        # Calculate edge statistics
        metrics['edge_strength_mean'].append(np.mean(edges))
        metrics['edge_strength_std'].append(np.std(edges))
        
        # Calculate corner statistics
        metrics['corner_response_mean'].append(np.mean(corners))
        metrics['corner_response_max'].append(np.max(corners))
        
        # Calculate adaptive threshold
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        threshold = edge_mean + 2 * edge_std
        metrics['threshold_ratio'].append(threshold / edge_mean if edge_mean > 0 else 0)
        
        # Create edge map and find contours
        edge_map = (edges > threshold).astype(np.uint8) * 255
        
        # Dilate
        dilate_kernel = np.ones((3, 3), dtype=np.float32)
        dilated = detector.conv2d(edge_map.astype(np.float32), dilate_kernel)
        dilated = (dilated > 0).astype(np.uint8) * 255
        
        # Find contours
        contours = detector.find_contours(dilated)
        metrics['contour_count'].append(len(contours))
        
        # Get prediction
        pred = detector.direct_regression(image)
        
        # Check detection success
        detection_success = pred[0] > 0.5
        metrics['detection_success'].append(detection_success)
        
        if detection_success:
            # Calculate IoU
            true_x, true_y, true_w, true_h = true_label[1], true_label[2], true_label[3], true_label[4]
            pred_x, pred_y, pred_w, pred_h = pred[1], pred[2], pred[3], pred[4]
            
            # Convert to box coordinates
            true_x1 = true_x - true_w/2
            true_y1 = true_y - true_h/2
            true_x2 = true_x + true_w/2
            true_y2 = true_y + true_h/2
            
            pred_x1 = pred_x - pred_w/2
            pred_y1 = pred_y - pred_h/2
            pred_x2 = pred_x + pred_w/2
            pred_y2 = pred_y + pred_h/2
            
            # Calculate intersection
            xi1 = max(true_x1, pred_x1)
            yi1 = max(true_y1, pred_y1)
            xi2 = min(true_x2, pred_x2)
            yi2 = min(true_y2, pred_y2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            true_area = true_w * true_h
            pred_area = pred_w * pred_h
            union_area = true_area + pred_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            metrics['iou_scores'].append(iou)
    
    # Calculate averages
    return {k: np.mean(v) if v else 0 for k, v in metrics.items()}


def visualize_kernel_responses():
    """Create detailed visualization of kernel responses for each dataset type."""
    
    detector = SimpleRectangleDetector()
    datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    fig, axes = plt.subplots(4, 6, figsize=(22, 16))
    fig.suptitle('Kernel Behavior Analysis: Filled vs Border-Only Rectangles\n(Including Laplacian Corner Detector)', fontsize=16, fontweight='bold')
    
    for row, dataset_type in enumerate(datasets):
        # Generate sample image
        image, label = create_noisy_image_with_rectangle(dataset_type=dataset_type)
        gray = np.mean(image, axis=2)
        
        # Apply kernels
        h_edges = detector.conv2d(gray, detector.h_kernel)
        v_edges = detector.conv2d(gray, detector.v_kernel)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Apply Laplacian corner detector
        corners = detector.conv2d(gray, detector.c_kernel)
        corners = np.abs(corners)  # Take absolute value
        
        # Threshold
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        threshold = edge_mean + 2 * edge_std
        edge_map = (edges > threshold).astype(np.uint8) * 255
        
        # Dilate
        dilate_kernel = np.ones((3, 3), dtype=np.float32)
        dilated = detector.conv2d(edge_map.astype(np.float32), dilate_kernel)
        dilated = (dilated > 0).astype(np.uint8) * 255
        
        # Plot
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f'{dataset_type}\nOriginal')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(h_edges, cmap='RdBu_r')
        axes[row, 1].set_title(f'Horizontal Edges\n(Sobel X)')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(v_edges, cmap='RdBu_r')
        axes[row, 2].set_title(f'Vertical Edges\n(Sobel Y)')
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(edges, cmap='hot')
        axes[row, 3].set_title(f'Combined Edges\nMean: {edge_mean:.1f}')
        axes[row, 3].axis('off')
        
        axes[row, 4].imshow(corners, cmap='hot')
        corner_max = np.max(corners)
        axes[row, 4].set_title(f'Laplacian Corners\nMax: {corner_max:.1f}')
        axes[row, 4].axis('off')
        
        axes[row, 5].imshow(dilated, cmap='gray')
        axes[row, 5].set_title(f'After Dilation\nThreshold: {threshold:.1f}')
        axes[row, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('kernel_behavior_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved kernel_behavior_analysis.png")
    plt.close()


def analyze_edge_profiles():
    """Analyze 1D edge profiles showing image with prediction and pixel axes."""
    
    detector = SimpleRectangleDetector()
    
    # Create figure with 4 rows (one per dataset) and 3 columns (image, horizontal profile, vertical profile)
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Edge Profile Analysis with Predictions', fontsize=16, fontweight='bold')
    
    datasets = [
        ('grayscale', 'Filled Grayscale'),
        ('grayscale_border', 'Border-Only Grayscale'),
        ('colored', 'Filled Colored'),
        ('colored_border', 'Border-Only Colored')
    ]
    
    for idx, (dataset_type, title) in enumerate(datasets):
        # Generate image and get prediction
        image, true_label = create_noisy_image_with_rectangle(dataset_type=dataset_type)
        gray = np.mean(image, axis=2)
        
        # Get model prediction
        pred = detector.direct_regression(image)
        pred_x, pred_y, pred_w, pred_h = pred[1], pred[2], pred[3], pred[4]
        
        # Get true rectangle coordinates
        true_x, true_y, true_w, true_h = true_label[1], true_label[2], true_label[3], true_label[4]
        img_h, img_w = gray.shape
        
        # Convert to pixel coordinates
        true_x1 = int((true_x - true_w/2) * img_w)
        true_y1 = int((true_y - true_h/2) * img_h)
        true_x2 = int((true_x + true_w/2) * img_w)
        true_y2 = int((true_y + true_h/2) * img_h)
        
        pred_x1 = int((pred_x - pred_w/2) * img_w) if pred[0] > 0.5 else 0
        pred_y1 = int((pred_y - pred_h/2) * img_h) if pred[0] > 0.5 else 0
        pred_x2 = int((pred_x + pred_w/2) * img_w) if pred[0] > 0.5 else 0
        pred_y2 = int((pred_y + pred_h/2) * img_h) if pred[0] > 0.5 else 0
        
        # Use true center for cross-sections (ground truth location)
        cy = (true_y1 + true_y2) // 2
        cx = (true_x1 + true_x2) // 2
        
        # Column 1: Image with prediction overlay
        ax_img = fig.add_subplot(gs[idx, 0])
        ax_img.imshow(image)
        
        # Draw true rectangle (green)
        from matplotlib.patches import Rectangle
        true_rect = Rectangle((true_x1, true_y1), true_x2-true_x1, true_y2-true_y1,
                              fill=False, edgecolor='lime', linewidth=2, linestyle='--',
                              label='Ground Truth')
        ax_img.add_patch(true_rect)
        
        # Draw predicted rectangle (red) if detected
        if pred[0] > 0.5:
            pred_rect = Rectangle((pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1,
                                  fill=False, edgecolor='red', linewidth=2,
                                  label='Prediction')
            ax_img.add_patch(pred_rect)
            iou_text = f"IoU: {calculate_iou(true_label, pred):.2f}"
        else:
            iou_text = "No detection"
        
        # Draw cross-section lines
        ax_img.axhline(y=cy, color='yellow', linestyle='-', alpha=0.5, linewidth=1)
        ax_img.axvline(x=cx, color='cyan', linestyle='-', alpha=0.5, linewidth=1)
        
        ax_img.set_title(f'{title}\n{iou_text}', fontsize=10)
        ax_img.set_xlabel('Pixel X')
        ax_img.set_ylabel('Pixel Y')
        ax_img.legend(loc='upper right', fontsize=7)
        
        # Extract cross-sections at true center
        h_profile = gray[cy, :]
        v_profile = gray[:, cx]
        
        # Apply edge detection
        h_edges = detector.conv2d(gray, detector.h_kernel)
        v_edges = detector.conv2d(gray, detector.v_kernel)
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Adjust for valid convolution (3x3 kernel reduces size by 2)
        offset = 1
        h_edge_profile = edges[cy - offset, :] if cy > 0 and cy < edges.shape[0] else np.zeros_like(h_profile)
        v_edge_profile = edges[:, cx - offset] if cx > 0 and cx < edges.shape[1] else np.zeros_like(v_profile)
        
        # Column 2: Horizontal cross-section
        ax_h = fig.add_subplot(gs[idx, 1])
        x_vals = np.arange(len(h_profile))
        ax_h.plot(x_vals, h_profile / 255.0, 'b-', label='Intensity', alpha=0.7, linewidth=1.5)
        if len(h_edge_profile) == len(x_vals) - 2:
            ax_h.plot(x_vals[1:-1], h_edge_profile / (np.max(edges) + 1e-8), 'r-',
                     label='Edge Response', linewidth=2)
        
        # Mark true and predicted boundaries
        ax_h.axvline(x=true_x1, color='lime', linestyle='--', alpha=0.7, label='True Edge')
        ax_h.axvline(x=true_x2, color='lime', linestyle='--', alpha=0.7)
        if pred[0] > 0.5:
            ax_h.axvline(x=pred_x1, color='red', linestyle='-', alpha=0.7, label='Predicted')
            ax_h.axvline(x=pred_x2, color='red', linestyle='-', alpha=0.7)
        ax_h.axvline(x=cx, color='yellow', linestyle='-', alpha=0.5)
        
        ax_h.set_title(f'Horizontal Cross-Section @ Y={cy}')
        ax_h.set_xlabel('Pixel X')
        ax_h.set_ylabel('Normalized Value')
        ax_h.set_xlim(0, img_w)
        ax_h.legend(loc='upper right', fontsize=7)
        ax_h.grid(True, alpha=0.3)
        
        # Column 3: Vertical cross-section
        ax_v = fig.add_subplot(gs[idx, 2])
        y_vals = np.arange(len(v_profile))
        ax_v.plot(v_profile / 255.0, y_vals, 'b-', label='Intensity', alpha=0.7, linewidth=1.5)
        if len(v_edge_profile) == len(y_vals) - 2:
            ax_v.plot(v_edge_profile / (np.max(edges) + 1e-8), y_vals[1:-1], 'r-',
                     label='Edge Response', linewidth=2)
        
        # Mark true and predicted boundaries
        ax_v.axhline(y=true_y1, color='lime', linestyle='--', alpha=0.7, label='True Edge')
        ax_v.axhline(y=true_y2, color='lime', linestyle='--', alpha=0.7)
        if pred[0] > 0.5:
            ax_v.axhline(y=pred_y1, color='red', linestyle='-', alpha=0.7, label='Predicted')
            ax_v.axhline(y=pred_y2, color='red', linestyle='-', alpha=0.7)
        ax_v.axhline(y=cy, color='cyan', linestyle='-', alpha=0.5)
        
        ax_v.set_title(f'Vertical Cross-Section @ X={cx}')
        ax_v.set_xlabel('Normalized Value')
        ax_v.set_ylabel('Pixel Y')
        ax_v.set_ylim(img_h, 0)  # Flip Y axis to match image
        ax_v.legend(loc='upper right', fontsize=7)
        ax_v.grid(True, alpha=0.3)
    
    plt.savefig('edge_profile_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved edge_profile_analysis.png")
    plt.close()


def calculate_iou(true_label, pred):
    """Calculate IoU between true and predicted boxes."""
    if pred[0] <= 0.5:
        return 0.0
    
    true_x, true_y, true_w, true_h = true_label[1], true_label[2], true_label[3], true_label[4]
    pred_x, pred_y, pred_w, pred_h = pred[1], pred[2], pred[3], pred[4]
    
    # Convert to box coordinates
    true_x1 = true_x - true_w/2
    true_y1 = true_y - true_h/2
    true_x2 = true_x + true_w/2
    true_y2 = true_y + true_h/2
    
    pred_x1 = pred_x - pred_w/2
    pred_y1 = pred_y - pred_h/2
    pred_x2 = pred_x + pred_w/2
    pred_y2 = pred_y + pred_h/2
    
    # Calculate intersection
    xi1 = max(true_x1, pred_x1)
    yi1 = max(true_y1, pred_y1)
    xi2 = min(true_x2, pred_x2)
    yi2 = min(true_y2, pred_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union_area = true_area + pred_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def generate_comparison_report():
    """Generate a detailed comparison report."""
    
    print("\n" + "="*80)
    print("KERNEL BEHAVIOR ANALYSIS: Why Border Datasets Outperform Filled Datasets")
    print("="*80)
    
    datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    results = {}
    
    print("\n📊 Running analysis on 20 samples per dataset...\n")
    
    for dataset_type in datasets:
        print(f"Analyzing {dataset_type}...")
        metrics = analyze_kernel_response(dataset_type, num_samples=20)
        results[dataset_type] = metrics
    
    # Print comparison table
    print("\n" + "-"*80)
    print("COMPARATIVE METRICS")
    print("-"*80)
    print(f"{'Metric':<25} {'Grayscale':<12} {'Color':<12} {'Gray-Border':<12} {'Color-Border':<12}")
    print("-"*80)
    
    metric_names = {
        'edge_strength_mean': 'Edge Strength (Mean)',
        'edge_strength_std': 'Edge Strength (Std)',
        'corner_response_mean': 'Corner Response (Mean)',
        'corner_response_max': 'Corner Response (Max)',
        'contour_count': 'Avg Contours Found',
        'threshold_ratio': 'Threshold Ratio',
        'detection_success': 'Detection Rate',
        'iou_scores': 'Mean IoU'
    }
    
    for metric_key, metric_name in metric_names.items():
        values = [results[ds][metric_key] for ds in datasets]
        print(f"{metric_name:<25} {values[0]:<12.3f} {values[1]:<12.3f} {values[2]:<12.3f} {values[3]:<12.3f}")
    
    print("-"*80)
    
    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("-"*80)
    
    gray_iou = results['grayscale']['iou_scores']
    gray_border_iou = results['grayscale_border']['iou_scores']
    color_iou = results['colored']['iou_scores']
    color_border_iou = results['colored_border']['iou_scores']
    
    print(f"\n1. PERFORMANCE GAPS:")
    print(f"   • Grayscale Border vs Filled:    {((gray_border_iou - gray_iou) / gray_iou * 100):+.1f}% improvement")
    print(f"   • Colored Border vs Filled:      {((color_border_iou - color_iou) / color_iou * 100):+.1f}% improvement")
    
    print(f"\n2. EDGE CHARACTERISTICS:")
    for ds in datasets:
        es_mean = results[ds]['edge_strength_mean']
        es_std = results[ds]['edge_strength_std']
        print(f"   • {ds:<20}: Mean={es_mean:.2f}, Std={es_std:.2f}, SNR={es_mean/es_std:.2f}")
    
    print(f"\n3. WHY BORDER DATASETS PERFORM BETTER:")
    print(f"   a) DOUBLE EDGE RESPONSE:")
    print(f"      - Border rectangles have 4 distinct edge transitions:")
    print(f"        * Background → Outer border edge (strong response)")
    print(f"        * Outer border edge → Inner border edge (strong response)")
    print(f"        * Inner border edge → Rectangle interior (strong response)")
    print(f"        * Rectangle interior → Inner border edge on exit")
    print(f"      - Filled rectangles only have 2 transitions (background→edge, edge→interior)")
    print(f"")
    print(f"   b) THRESHOLD ROBUSTNESS:")
    print(f"      - Border datasets have more consistent edge strengths")
    print(f"      - Lower variance means adaptive threshold works better")
    print(f"")
    print(f"   c) CONTOUR QUALITY:")
    print(f"      - Border rectangles create cleaner, closed contours")
    print(f"      - Filled rectangles may have 'filled' noise affecting edges")
    print(f"")
    print(f"   d) DILATION EFFECTS:")
    print(f"      - Border edges dilate into a 'band' that's easier to contour")
    print(f"      - Filled rectangle edges may dilate inconsistently")
    
    print(f"\n4. KERNEL BEHAVIOR EXPLANATION:")
    print(f"   The Sobel kernels (3x3) compute gradients using:")
    print(f"   • Horizontal kernel: [-1 0 1; -2 0 2; -1 0 1]")
    print(f"   • Vertical kernel:   [-1 -2 -1; 0 0 0; 1 2 1]")
    print(f"")
    print(f"   For a border rectangle (thickness=3):")
    print(f"   - The kernel 'sees' multiple transitions in its 3x3 window")
    print(f"   - Each border edge produces a strong gradient response")
    print(f"   - The interior of a filled rectangle is uniform (no gradient)")
    print(f"   - But the border-only has TWO edges per side (4 total strong responses)")
    
    print(f"\n5. LAPLACIAN CORNER DETECTOR ANALYSIS:")
    print(f"   The Laplacian kernel (3x3) detects corners and rapid intensity changes:")
    print(f"   • Laplacian kernel: [0 -1 0; -1 4 -1; 0 -1 0]")
    print(f"   • This is a second-derivative operator (measures rate of change of gradient)")
    print(f"   • Strong response at corners where edges meet at 90° angles")
    print(f"")
    print(f"   Corner Response by Dataset:")
    for ds in datasets:
        corner_mean = results[ds]['corner_response_mean']
        corner_max = results[ds]['corner_response_max']
        print(f"   • {ds:<20}: Mean={corner_mean:.2f}, Max={corner_max:.2f}")
    print(f"")
    print(f"   Why Border Rectangles Have Stronger Corner Response:")
    print(f"   - Border rectangles have 4 distinct corner regions (inner + outer corners)")
    print(f"   - Each corner produces a strong Laplacian response due to 90° edge junctions")
    print(f"   - Filled rectangles have corners but edges fade into uniform interior")
    print(f"   - The border's 'hole' creates additional corner-like transitions")
    print(f"")
    print(f"   Corner Detection Impact on Rectangle Detection:")
    print(f"   - Strong corner responses help validate rectangle presence")
    print(f"   - 4 corners = geometric proof of rectangular shape")
    print(f"   - Border rectangles have 8 strong corner signals (4 outer + 4 inner)")
    print(f"   - This reinforces the edge-based detection")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    print("🔄 Generating kernel behavior visualizations...")
    visualize_kernel_responses()
    
    print("\n🔄 Generating edge profile analysis...")
    analyze_edge_profiles()
    
    print("\n🔄 Running comparative metrics...")
    results = generate_comparison_report()
    
    print("\n✅ Analysis complete! Generated files:")
    print("   • kernel_behavior_analysis.png - Full kernel response visualization")
    print("   • edge_profile_analysis.png - Cross-section edge profiles")
