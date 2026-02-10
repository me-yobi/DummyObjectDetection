#!/usr/bin/env python3
"""Evaluate the regression-based detector on validation set"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .config import Config
from .models.detector import SimpleRectangleDetector
from .data.dataset import RectangleDataset
import matplotlib.pyplot as plt

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

def evaluate_model():
    """Evaluate the regression model on validation set"""
    cfg = Config()
    
    # Create model (no training needed)
    model = SimpleRectangleDetector()
    print("Created regression-based model (no training required)")
    
    # Get all data (no split needed since we're not training)
    full_dataset = RectangleDataset(cfg.DATASET_DIR)
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Evaluate on entire dataset
    model.eval()
    
    all_predictions = []
    all_targets = []
    ious = []
    losses = []
    
    print("\nEvaluating on all samples...")
    
    # Process in batches for memory efficiency
    batch_size = 32
    pbar = tqdm(range(0, len(full_dataset), batch_size), desc="Evaluating")
    for i in pbar:
        batch_end = min(i + batch_size, len(full_dataset))
        batch_indices = range(i, batch_end)
        
        batch_predictions = []
        batch_losses = []
        
        for idx in batch_indices:
            image, target = full_dataset[idx]
            
            # Convert from CxHxW to HxWxC
            img = image.transpose(1, 2, 0)
            
            # Forward pass (direct regression)
            output = model.direct_regression(img)
            
            # Calculate loss
            loss = np.mean((output - target) ** 2)
            
            batch_predictions.append(output)
            batch_losses.append(loss)
            all_targets.append(target)
            
            # Calculate IoU
            iou = calculate_iou(output, target)
            ious.append(iou)
        
        all_predictions.extend(batch_predictions)
        losses.extend(batch_losses)
        
        # Update progress with current metrics
        if len(losses) > 0:
            current_loss = np.mean(losses[-batch_size:])
            current_iou = np.mean(ious[-batch_size:])
            pbar.set_description(f"Evaluating (loss: {current_loss:.4f}, iou: {current_iou:.3f})")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    ious = np.array(ious)
    losses = np.array(losses)
    
    # Print statistics
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(all_predictions)}")
    print(f"\nLoss (MSE):")
    print(f"  Mean: {np.mean(losses):.6f}")
    print(f"  Std:  {np.std(losses):.6f}")
    
    print(f"\nIoU (Intersection over Union):")
    print(f"  Mean: {np.mean(ious):.4f}")
    print(f"  Std:  {np.std(ious):.4f}")
    print(f"  Min:  {np.min(ious):.4f}")
    print(f"  Max:  {np.max(ious):.4f}")
    print(f"  Samples with IoU > 0.5: {np.sum(ious > 0.5)} / {len(ious)} ({100*np.mean(ious > 0.5):.1f}%)")
    print(f"  Samples with IoU > 0.7: {np.sum(ious > 0.7)} / {len(ious)} ({100*np.mean(ious > 0.7):.1f}%)")
    
    # Overall metrics using scikit-learn
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    
    print(f"\nOverall Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Visualize some predictions
    visualize_predictions(model, full_dataset, all_predictions, all_targets, num_samples=10)
    
    return all_predictions, all_targets, ious, losses

def visualize_predictions(model, dataset, predictions, targets, num_samples=10):
    """Visualize model predictions"""
    print(f"\nVisualizing {num_samples} predictions...")
    
    plt.figure(figsize=(20, 12))
    
    for i in range(min(num_samples, len(dataset))):
        image, true_target = dataset[i]
        pred_target = predictions[i]
        
        # Convert from CxHxW to HxWxC
        img = image.transpose(1, 2, 0)
        
        # Denormalize image for display
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        
        # Convert normalized coordinates to pixel coordinates
        h, w = img.shape[:2]
        
        # True box (green)
        true_x = true_target[1] * w
        true_y = true_target[2] * h
        true_w = true_target[3] * w
        true_h = true_target[4] * h
        true_x1 = true_x - true_w/2
        true_y1 = true_y - true_h/2
        
        # Predicted box (red)
        pred_x = pred_target[1] * w
        pred_y = pred_target[2] * h
        pred_w = pred_target[3] * w
        pred_h = pred_target[4] * h
        pred_x1 = pred_x - pred_w/2
        pred_y1 = pred_y - pred_h/2
        
        # Draw true box
        true_rect = plt.Rectangle((true_x1, true_y1), true_w, true_h, 
                              fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(true_rect)
        
        # Draw predicted box
        pred_rect = plt.Rectangle((pred_x1, pred_y1), pred_w, pred_h, 
                              fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(pred_rect)
        
        # Calculate IoU for this sample
        iou = calculate_iou(pred_target, true_target)
        plt.title(f'Sample {i+1}\nIoU: {iou:.3f}')
        plt.axis('off')
    
    plt.suptitle('Model Predictions (Green=True, Red=Predicted)', fontsize=16)
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization to: regression_predictions.png")

def main():
    """Main function"""
    print("=== Rectangle Detection with Direct Regression ===")
    print("No training required - direct computation from convolution features!")
    
    predictions, targets, ious, losses = evaluate_model()
    
    print("\n=== Summary ===")
    print("✓ Direct regression approach (no training)")
    print("✓ Uses convolution features to directly compute bounding box")
    print("✓ Fast inference - no neural network weights to optimize")
    print(f"✓ Mean IoU: {np.mean(ious):.3f}")
    print(f"✓ MSE: {np.mean(losses):.6f}")

if __name__ == "__main__":
    main()
