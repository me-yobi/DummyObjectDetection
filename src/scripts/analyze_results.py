#!/usr/bin/env python3
"""Analyze model performance on validation set without PyTorch"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .config import Config
from ..data.dataset import RectangleDataset
from ..models.detector import SimpleRectangleDetector

def calculate_iou(pred_box, true_box):
    """Calculate Intersection over Union (IoU) between two boxes"""
    # Convert from normalized [x_center, y_center, width, height] to [x1, y1, x2, y2]
    pred_x1 = pred_box[1] - pred_box[3] / 2
    pred_y1 = pred_box[2] - pred_box[4] / 2
    pred_x2 = pred_box[1] + pred_box[3] / 2
    pred_y2 = pred_box[2] + pred_box[4] / 2
    
    true_x1 = true_box[1] - true_box[3] / 2
    true_y1 = true_box[2] - true_box[4] / 2
    true_x2 = true_box[1] + true_box[3] / 2
    true_y2 = true_box[2] + true_box[4] / 2
    
    # Calculate intersection
    x1 = max(pred_x1, true_x1)
    y1 = max(pred_y1, true_y1)
    x2 = min(pred_x2, true_x2)
    y2 = min(pred_y2, true_y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    pred_area = pred_box[3] * pred_box[4]
    true_area = true_box[3] * true_box[4]
    union = pred_area + true_area - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_model():
    """Analyze model performance on validation set"""
    cfg = Config()
    
    # Load analytic model (no trainable weights)
    model = SimpleRectangleDetector()
    
    # Load validation dataset
    val_dataset = RectangleDataset(
        cfg.DATASET_DIR,
        normalize=False  # Don't normalize for visualization
    )
    
    print(f"\nAnalyzing on {len(val_dataset)} validation samples...")
    
    # Metrics
    ious = []
    center_errors = []
    size_errors = []
    all_predictions = []
    all_targets = []
    
    # Analyze first 10 samples with visualization
    num_samples_to_show = min(10, len(val_dataset))
    
    plt.figure(figsize=(20, 12))
    for i in range(num_samples_to_show):
        image, true_label = val_dataset[i]
        
        # Get prediction
        model.eval()
        # Convert from CHW to HWC
        img_hwc = image.transpose(1, 2, 0)
        pred = model.forward(img_hwc)
        
        # Calculate metrics
        iou = calculate_iou(pred, true_label)
        center_error = np.sqrt((pred[1] - true_label[1])**2 + (pred[2] - true_label[2])**2)
        size_error = np.sqrt((pred[3] - true_label[3])**2 + (pred[4] - true_label[4])**2)
        
        ious.append(iou)
        center_errors.append(center_error)
        size_errors.append(size_error)
        all_predictions.append(pred)
        all_targets.append(true_label)
        
        # Plot
        plt.subplot(2, 5, i + 1)
        img_np = image.transpose(1, 2, 0)
        
        # Denormalize image if needed
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        plt.imshow(img_np)
        
        # Draw true box (green) and predicted box (red)
        true_box = true_label[1:] * 256  # Denormalize
        pred_box = pred[1:] * 256
        
        # True box
        true_x1 = true_box[0] - true_box[2] / 2
        true_y1 = true_box[1] - true_box[3] / 2
        plt.gca().add_patch(plt.Rectangle((true_x1, true_y1), true_box[2], true_box[3], 
                                         fill=False, edgecolor='green', linewidth=2, label='True'))
        
        # Predicted box
        pred_x1 = pred_box[0] - pred_box[2] / 2
        pred_y1 = pred_box[1] - pred_box[3] / 2
        plt.gca().add_patch(plt.Rectangle((pred_x1, pred_y1), pred_box[2], pred_box[3], 
                                         fill=False, edgecolor='red', linewidth=2, label='Pred'))
        
        plt.title(f'Sample {i+1}\nIoU: {iou:.3f}')
        plt.axis('off')
    
    plt.suptitle('Validation Predictions (Green=True, Red=Predicted)', fontsize=16)
    plt.tight_layout()
    plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analyze entire validation set
    print("\nAnalyzing entire validation set...")
    model.eval()
    for i in range(len(val_dataset)):
        image, true_label = val_dataset[i]
        img_hwc = image.transpose(1, 2, 0)
        pred = model.forward(img_hwc)
        
        ious.append(calculate_iou(pred, true_label))
        center_errors.append(np.sqrt((pred[1] - true_label[1])**2 + (pred[2] - true_label[2])**2))
        size_errors.append(np.sqrt((pred[3] - true_label[3])**2 + (pred[4] - true_label[4])**2))
        all_predictions.append(pred)
        all_targets.append(true_label)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Print statistics
    print("\n=== Model Performance Analysis ===")
    print(f"Total samples evaluated: {len(ious)}")
    print(f"\nIoU (Intersection over Union):")
    print(f"  Mean: {np.mean(ious):.4f}")
    print(f"  Std:  {np.std(ious):.4f}")
    print(f"  Min:  {np.min(ious):.4f}")
    print(f"  Max:  {np.max(ious):.4f}")
    print(f"  Samples with IoU > 0.5: {np.sum(np.array(ious) > 0.5)} / {len(ious)} ({100*np.mean(np.array(ious) > 0.5):.1f}%)")
    
    print(f"\nCenter Position Error (normalized):")
    print(f"  Mean: {np.mean(center_errors):.4f}")
    print(f"  Std:  {np.std(center_errors):.4f}")
    
    print(f"\nSize Error (normalized):")
    print(f"  Mean: {np.mean(size_errors):.4f}")
    print(f"  Std:  {np.std(size_errors):.4f}")
    
    # Calculate overall metrics using scikit-learn
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    
    print(f"\nOverall Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Plot metrics distribution
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(ious, bins=20, alpha=0.7, color='blue')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f'IoU Distribution\n(Mean: {np.mean(ious):.3f})')
    plt.axvline(np.mean(ious), color='red', linestyle='--', label='Mean')
    plt.axvline(0.5, color='green', linestyle='--', label='0.5 threshold')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(center_errors, bins=20, alpha=0.7, color='orange')
    plt.xlabel('Center Error')
    plt.ylabel('Count')
    plt.title(f'Center Error Distribution\n(Mean: {np.mean(center_errors):.3f})')
    plt.axvline(np.mean(center_errors), color='red', linestyle='--', label='Mean')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(size_errors, bins=20, alpha=0.7, color='purple')
    plt.xlabel('Size Error')
    plt.ylabel('Count')
    plt.title(f'Size Error Distribution\n(Mean: {np.mean(size_errors):.3f})')
    plt.axvline(np.mean(size_errors), color='red', linestyle='--', label='Mean')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization files saved:")
    print("  - validation_predictions.png: Visual comparison of predictions")
    print("  - metrics_distribution.png: Distribution of performance metrics")

if __name__ == "__main__":
    analyze_model()
