#!/usr/bin/env python3
"""
Quantitative Kernel Effectiveness Analysis

Measures how effective each kernel is at detecting rectangle edges and corners.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.DataGenerator import create_noisy_image_with_rectangle
from src.models.detector import SimpleRectangleDetector


def evaluate_kernel_effectiveness(num_samples=50):
    """
    Quantitatively evaluate each kernel's effectiveness.
    Returns metrics for h_kernel, v_kernel, and c_kernel.
    """
    detector = SimpleRectangleDetector()
    datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    # Metrics to collect
    results = {ds: {
        'h_kernel': {'edge_precision': [], 'edge_recall': [], 'snr': [], 'peak_response': []},
        'v_kernel': {'edge_precision': [], 'edge_recall': [], 'snr': [], 'peak_response': []},
        'c_kernel': {'corner_precision': [], 'corner_recall': [], 'snr': [], 'peak_response': []},
        'combined': {'iou': [], 'detection_rate': []}
    } for ds in datasets}
    
    print(f"🔄 Analyzing {num_samples} samples per dataset...")
    
    for dataset_type in datasets:
        print(f"  Processing {dataset_type}...")
        
        for i in range(num_samples):
            # Generate image
            image, true_label = create_noisy_image_with_rectangle(dataset_type=dataset_type)
            gray = np.mean(image, axis=2)
            img_h, img_w = gray.shape
            
            # Get true rectangle coordinates in pixels
            true_x, true_y, true_w, true_h = true_label[1], true_label[2], true_label[3], true_label[4]
            true_x1 = int((true_x - true_w/2) * img_w)
            true_y1 = int((true_y - true_h/2) * img_h)
            true_x2 = int((true_x + true_w/2) * img_w)
            true_y2 = int((true_y + true_h/2) * img_h)
            
            # Apply kernels
            h_response = np.abs(detector.conv2d(gray, detector.h_kernel))
            v_response = np.abs(detector.conv2d(gray, detector.v_kernel))
            c_response = np.abs(detector.conv2d(gray, detector.c_kernel))
            
            # Create edge ground truth masks
            h_edge_mask = np.zeros_like(gray, dtype=bool)
            v_edge_mask = np.zeros_like(gray, dtype=bool)
            corner_mask = np.zeros_like(gray, dtype=bool)
            
            # Mark true horizontal edges (top and bottom of rectangle)
            edge_thickness = 3
            for dy in range(-edge_thickness, edge_thickness+1):
                y_top = max(0, min(img_h-1, true_y1 + dy))
                y_bottom = max(0, min(img_h-1, true_y2 + dy))
                if 0 <= y_top < h_response.shape[0]:
                    h_edge_mask[y_top, max(0,true_x1):min(img_w,true_x2+1)] = True
                if 0 <= y_bottom < h_response.shape[0]:
                    h_edge_mask[y_bottom, max(0,true_x1):min(img_w,true_x2+1)] = True
            
            # Mark true vertical edges (left and right of rectangle)
            for dx in range(-edge_thickness, edge_thickness+1):
                x_left = max(0, min(img_w-1, true_x1 + dx))
                x_right = max(0, min(img_w-1, true_x2 + dx))
                if 0 <= x_left < v_response.shape[1]:
                    v_edge_mask[max(0,true_y1):min(img_h,true_y2+1), x_left] = True
                if 0 <= x_right < v_response.shape[1]:
                    v_edge_mask[max(0,true_y1):min(img_h,true_y2+1), x_right] = True
            
            # Mark true corners (4 corners of rectangle)
            corners = [
                (true_x1, true_y1), (true_x2, true_y1),
                (true_x1, true_y2), (true_x2, true_y2)
            ]
            corner_thickness = 5
            for cx, cy in corners:
                for dy in range(-corner_thickness, corner_thickness+1):
                    for dx in range(-corner_thickness, corner_thickness+1):
                        py = max(0, min(c_response.shape[0]-1, cy + dy))
                        px = max(0, min(c_response.shape[1]-1, cx + dx))
                        corner_mask[py, px] = True
            
            # Adjust masks for valid convolution output
            h_edge_mask = h_edge_mask[1:-1, 1:-1] if h_edge_mask.shape[0] > 2 else h_edge_mask
            v_edge_mask = v_edge_mask[1:-1, 1:-1] if v_edge_mask.shape[0] > 2 else v_edge_mask
            corner_mask = corner_mask[1:-1, 1:-1] if corner_mask.shape[0] > 2 else corner_mask
            
            # Calculate metrics for each kernel
            for kernel_name, response, mask in [
                ('h_kernel', h_response, h_edge_mask),
                ('v_kernel', v_response, v_edge_mask),
                ('c_kernel', c_response, corner_mask)
            ]:
                # Ensure shapes match
                min_h = min(response.shape[0], mask.shape[0])
                min_w = min(response.shape[1], mask.shape[1])
                response = response[:min_h, :min_w]
                mask = mask[:min_h, :min_w]
                
                # Threshold response
                threshold = np.mean(response) + 2 * np.std(response)
                detected = response > threshold
                
                # True positives: detected AND is edge
                tp = np.sum(detected & mask)
                # False positives: detected but NOT edge
                fp = np.sum(detected & ~mask)
                # False negatives: NOT detected but IS edge
                fn = np.sum(~detected & mask)
                
                # Precision and Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Signal-to-Noise Ratio (mean response on edges vs background)
                edge_response = np.mean(response[mask]) if np.any(mask) else 0
                bg_response = np.mean(response[~mask]) if np.any(~mask) else 1
                snr = edge_response / (bg_response + 1e-8)
                
                # Peak response
                peak = np.max(response[mask]) if np.any(mask) else 0
                
                metric_type = 'corner_precision' if kernel_name == 'c_kernel' else 'edge_precision'
                metric_recall = 'corner_recall' if kernel_name == 'c_kernel' else 'edge_recall'
                
                results[dataset_type][kernel_name][metric_type].append(precision)
                results[dataset_type][kernel_name][metric_recall].append(recall)
                results[dataset_type][kernel_name]['snr'].append(snr)
                results[dataset_type][kernel_name]['peak_response'].append(peak)
            
            # Get model prediction for combined effectiveness
            pred = detector.direct_regression(image)
            results[dataset_type]['combined']['detection_rate'].append(1.0 if pred[0] > 0.5 else 0.0)
            
            if pred[0] > 0.5:
                # Calculate IoU
                pred_x, pred_y, pred_w, pred_h = pred[1], pred[2], pred[3], pred[4]
                true_area = true_w * true_h
                pred_area = pred_w * pred_h
                
                xi1 = max(true_x - true_w/2, pred_x - pred_w/2)
                yi1 = max(true_y - true_h/2, pred_y - pred_h/2)
                xi2 = min(true_x + true_w/2, pred_x + pred_w/2)
                yi2 = min(true_y + true_h/2, pred_y + pred_h/2)
                
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union_area = true_area + pred_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                results[dataset_type]['combined']['iou'].append(iou)
    
    return results


def print_quantitative_report(results):
    """Print quantitative analysis report."""
    
    print("\n" + "="*100)
    print("QUANTITATIVE KERNEL EFFECTIVENESS ANALYSIS")
    print("="*100)
    
    datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    # Aggregate results
    print("\n📊 KERNEL PERFORMANCE METRICS (averaged across all datasets)")
    print("-"*100)
    
    for kernel_name, kernel_display in [
        ('h_kernel', 'Horizontal Edge (Sobel X)'),
        ('v_kernel', 'Vertical Edge (Sobel Y)'),
        ('c_kernel', 'Corner Detector (Laplacian)')
    ]:
        print(f"\n🔍 {kernel_display}")
        print("-"*100)
        print(f"{'Dataset':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'SNR':<12} {'Peak Response':<15}")
        print("-"*100)
        
        for ds in datasets:
            precision = np.mean(results[ds][kernel_name]['edge_precision' if kernel_name != 'c_kernel' else 'corner_precision'])
            recall = np.mean(results[ds][kernel_name]['edge_recall' if kernel_name != 'c_kernel' else 'corner_recall'])
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            snr = np.mean(results[ds][kernel_name]['snr'])
            peak = np.mean(results[ds][kernel_name]['peak_response'])
            
            print(f"{ds:<20} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {snr:<12.2f} {peak:<15.1f}")
    
    print("\n" + "="*100)
    print("📈 COMBINED DETECTION PERFORMANCE")
    print("-"*100)
    print(f"{'Dataset':<25} {'Detection Rate':<20} {'Mean IoU':<15}")
    print("-"*100)
    
    for ds in datasets:
        det_rate = np.mean(results[ds]['combined']['detection_rate']) * 100
        mean_iou = np.mean(results[ds]['combined']['iou']) if results[ds]['combined']['iou'] else 0
        print(f"{ds:<25} {det_rate:<20.1f}% {mean_iou:<15.3f}")
    
    print("\n" + "="*100)
    print("🔬 KEY QUANTITATIVE FINDINGS")
    print("-"*100)
    
    # Calculate aggregate scores
    h_f1_scores = []
    v_f1_scores = []
    c_f1_scores = []
    
    for ds in datasets:
        h_prec = np.mean(results[ds]['h_kernel']['edge_precision'])
        h_rec = np.mean(results[ds]['h_kernel']['edge_recall'])
        h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0
        h_f1_scores.append(h_f1)
        
        v_prec = np.mean(results[ds]['v_kernel']['edge_precision'])
        v_rec = np.mean(results[ds]['v_kernel']['edge_recall'])
        v_f1 = 2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0
        v_f1_scores.append(v_f1)
        
        c_prec = np.mean(results[ds]['c_kernel']['corner_precision'])
        c_rec = np.mean(results[ds]['c_kernel']['corner_recall'])
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
        c_f1_scores.append(c_f1)
    
    print(f"\n1. KERNEL EFFECTIVENESS RANKING (by F1-Score):")
    print(f"   • Horizontal Edge (Sobel X): {np.mean(h_f1_scores):.3f} ± {np.std(h_f1_scores):.3f}")
    print(f"   • Vertical Edge (Sobel Y):   {np.mean(v_f1_scores):.3f} ± {np.std(v_f1_scores):.3f}")
    print(f"   • Corner (Laplacian):        {np.mean(c_f1_scores):.3f} ± {np.std(c_f1_scores):.3f}")
    
    print(f"\n2. BEST PERFORMING KERNEL BY DATASET:")
    for i, ds in enumerate(datasets):
        scores = [('Horizontal', h_f1_scores[i]), ('Vertical', v_f1_scores[i]), ('Corner', c_f1_scores[i])]
        best = max(scores, key=lambda x: x[1])
        print(f"   • {ds:<20}: {best[0]} Edge Detector (F1={best[1]:.3f})")
    
    print(f"\n3. SNR ANALYSIS (Signal-to-Noise Ratio):")
    for kernel_name, name in [('h_kernel', 'Horizontal'), ('v_kernel', 'Vertical'), ('c_kernel', 'Corner')]:
        snrs = [np.mean(results[ds][kernel_name]['snr']) for ds in datasets]
        print(f"   • {name}: Mean SNR = {np.mean(snrs):.2f} ± {np.std(snrs):.2f}")
    
    print(f"\n4. DATASET PERFORMANCE CORRELATION:")
    border_datasets = ['grayscale_border', 'colored_border']
    filled_datasets = ['grayscale', 'colored']
    
    border_f1 = np.mean([h_f1_scores[i] + v_f1_scores[i] + c_f1_scores[i] for i in [2, 3]]) / 3
    filled_f1 = np.mean([h_f1_scores[i] + v_f1_scores[i] + c_f1_scores[i] for i in [0, 1]]) / 3
    
    print(f"   • Border datasets average F1:  {border_f1:.3f}")
    print(f"   • Filled datasets average F1:  {filled_f1:.3f}")
    print(f"   • Improvement with borders:    {((border_f1 - filled_f1) / filled_f1 * 100):+.1f}%")
    
    print("\n" + "="*100)
    
    return {
        'h_f1': np.mean(h_f1_scores),
        'v_f1': np.mean(v_f1_scores),
        'c_f1': np.mean(c_f1_scores)
    }


def visualize_kernel_effectiveness(results):
    """Create visualizations showing kernel effectiveness."""
    
    datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    dataset_labels = ['Gray\n(Filled)', 'Color\n(Filled)', 'Gray\n(Border)', 'Color\n(Border)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantitative Kernel Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: F1-Scores by kernel and dataset
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.25
    
    h_f1 = []
    v_f1 = []
    c_f1 = []
    
    for ds in datasets:
        h_prec = np.mean(results[ds]['h_kernel']['edge_precision'])
        h_rec = np.mean(results[ds]['h_kernel']['edge_recall'])
        h_f1.append(2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0)
        
        v_prec = np.mean(results[ds]['v_kernel']['edge_precision'])
        v_rec = np.mean(results[ds]['v_kernel']['edge_recall'])
        v_f1.append(2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0)
        
        c_prec = np.mean(results[ds]['c_kernel']['corner_precision'])
        c_rec = np.mean(results[ds]['c_kernel']['corner_recall'])
        c_f1.append(2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0)
    
    ax1.bar(x - width, h_f1, width, label='Horizontal Edge', color='skyblue')
    ax1.bar(x, v_f1, width, label='Vertical Edge', color='lightcoral')
    ax1.bar(x + width, c_f1, width, label='Corner', color='lightgreen')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Kernel Detection F1-Score by Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Signal-to-Noise Ratio
    ax2 = axes[0, 1]
    h_snr = [np.mean(results[ds]['h_kernel']['snr']) for ds in datasets]
    v_snr = [np.mean(results[ds]['v_kernel']['snr']) for ds in datasets]
    c_snr = [np.mean(results[ds]['c_kernel']['snr']) for ds in datasets]
    
    ax2.bar(x - width, h_snr, width, label='Horizontal Edge', color='skyblue')
    ax2.bar(x, v_snr, width, label='Vertical Edge', color='lightcoral')
    ax2.bar(x + width, c_snr, width, label='Corner', color='lightgreen')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('Kernel SNR by Dataset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision vs Recall scatter
    ax3 = axes[1, 0]
    colors = {'h_kernel': 'blue', 'v_kernel': 'red', 'c_kernel': 'green'}
    markers = {'grayscale': 'o', 'colored': 's', 'grayscale_border': '^', 'colored_border': 'D'}
    
    for ds in datasets:
        for kernel in ['h_kernel', 'v_kernel', 'c_kernel']:
            metric_type = 'corner' if kernel == 'c_kernel' else 'edge'
            prec = np.mean(results[ds][kernel][f'{metric_type}_precision'])
            rec = np.mean(results[ds][kernel][f'{metric_type}_recall'])
            ax3.scatter(rec, prec, c=colors[kernel], marker=markers[ds], s=100, alpha=0.7,
                       label=f"{kernel.split('_')[0]} - {ds[:5]}")
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall by Kernel and Dataset')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Horizontal'),
        Patch(facecolor='red', label='Vertical'),
        Patch(facecolor='green', label='Corner')
    ]
    ax3.legend(handles=legend_elements, loc='lower left')
    
    # Plot 4: Overall Detection Performance
    ax4 = axes[1, 1]
    det_rates = [np.mean(results[ds]['combined']['detection_rate']) * 100 for ds in datasets]
    mean_ious = [np.mean(results[ds]['combined']['iou']) * 100 if results[ds]['combined']['iou'] else 0 for ds in datasets]
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, det_rates, width, label='Detection Rate', color='steelblue', alpha=0.8)
    bars2 = ax4_twin.bar(x + width/2, mean_ious, width, label='Mean IoU', color='orange', alpha=0.8)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Detection Rate (%)', color='steelblue')
    ax4_twin.set_ylabel('Mean IoU (%)', color='orange')
    ax4.set_title('Overall Detection Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dataset_labels)
    ax4.set_ylim(0, 110)
    ax4_twin.set_ylim(0, 110)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('kernel_quantitative_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved kernel_quantitative_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("🔄 Running quantitative kernel effectiveness analysis...")
    results = evaluate_kernel_effectiveness(num_samples=50)
    
    print("\n🔄 Generating visualizations...")
    visualize_kernel_effectiveness(results)
    
    kernel_scores = print_quantitative_report(results)
    
    print("\n✅ Analysis complete!")
    print("\nGenerated files:")
    print("  • kernel_quantitative_analysis.png - Visualization of kernel effectiveness")
