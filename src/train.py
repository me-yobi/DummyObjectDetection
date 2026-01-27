#!/usr/bin/env python3
"""Evaluate the regression-based detector on validation set"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .config import Config
from .models.detector import SimpleRectangleDetector
from .data.dataset import get_dataloaders
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
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        cfg.DATASET_DIR,
        batch_size=32,
        val_split=0.2,
        random_state=42
    )
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Evaluate on validation set
    model.eval()
    
    all_predictions = []
    all_targets = []
    ious = []
    losses = []
    
    print("\nEvaluating on validation set...")
    with tqdm(val_loader, desc="Evaluating") as pbar:
        for batch_idx, (images, targets) in enumerate(pbar):
            # Convert to numpy if needed
            if hasattr(images, 'numpy'):
                images = images.numpy()
            if hasattr(targets, 'numpy'):
                targets = targets.numpy()
            
            batch_predictions = []
            batch_losses = []
            
            for i in range(images.shape[0]):
                # Convert from CxHxW to HxWxC
                img = images[i].transpose(1, 2, 0)
                target = targets[i]
                
                # Forward pass (direct regression)
                output = model.direct_regression(img)
                
                # Calculate loss
                loss = np.mean((output - target) ** 2)
                
                batch_predictions.append(output)
                batch_losses.append(loss)
            
            batch_predictions = np.array(batch_predictions)
            batch_targets = targets
            
            # Calculate IoU for each sample
            for pred, true in zip(batch_predictions, batch_targets):
                iou = calculate_iou(pred, true)
                ious.append(iou)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            losses.extend(batch_losses)
            
            # Update progress bar
            avg_loss = np.mean(batch_losses)
            avg_iou = np.mean([calculate_iou(p, t) for p, t in zip(batch_predictions, batch_targets)])
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'iou': f'{avg_iou:.3f}'})
    
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
    visualize_predictions(model, val_loader, num_samples=10)
    
    return all_predictions, all_targets, ious, losses

def visualize_predictions(model, val_loader, num_samples=10):
    """Visualize model predictions"""
    print(f"\nVisualizing {num_samples} predictions...")
    
    plt.figure(figsize=(20, 12))
    samples_shown = 0
    
    for batch_idx, (images, targets) in enumerate(val_loader):
        if samples_shown >= num_samples:
            break
        
        # Convert to numpy if needed
        if hasattr(images, 'numpy'):
            images = images.numpy()
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
        
        for i in range(min(images.shape[0], num_samples - samples_shown)):
            # Convert from CxHxW to HxWxC
            img = images[i].transpose(1, 2, 0)
            target = targets[i]
            
            # Get prediction
            pred = model.direct_regression(img)
            
            # Calculate IoU
            iou = calculate_iou(pred, target)
            
            # Plot
            plt.subplot(2, 5, samples_shown + 1)
            
            # Denormalize image for display
            img_display = (img * 255).astype(np.uint8)
            
            plt.imshow(img_display)
            
            # Draw true box (green) and predicted box (red)
            true_box = target[1:] * 256  # Denormalize
            pred_box = pred[1:] * 256
            
            # True box
            true_x1 = true_box[0] - true_box[2] / 2
            true_y1 = true_box[1] - true_box[3] / 2
            plt.gca().add_patch(plt.Rectangle((true_x1, true_y1), true_box[2], true_box[3], 
                                             fill=False, edgecolor='green', linewidth=2, label='True'))
            plt.text(true_x1, true_y1 - 10, 'GT', color='green', fontsize=8)
            
            # Predicted box
            pred_x1 = pred_box[0] - pred_box[2] / 2
            pred_y1 = pred_box[1] - pred_box[3] / 2
            plt.gca().add_patch(plt.Rectangle((pred_x1, pred_y1), pred_box[2], pred_box[3], 
                                             fill=False, edgecolor='red', linewidth=3, label='Pred'))
            plt.text(pred_x1, pred_y1 - 10, 'PRED', color='red', fontsize=8)
            
            plt.title(f'Sample {samples_shown+1}\nIoU: {iou:.3f}')
            plt.axis('off')
            
            samples_shown += 1
    
    plt.suptitle('Regression Predictions (Green=True, Red=Predicted)', fontsize=16)
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to: regression_predictions.png")

def main():
    """Main evaluation function"""
    print("=== Rectangle Detection with Direct Regression ===")
    print("No training required - direct computation from convolution features!\n")
    
    # Check if dataset exists
    if not os.path.exists(Config.DATASET_DIR):
        print(f"Dataset not found at {Config.DATASET_DIR}")
        print("Please run: python prepare_data.py")
        return
    
    # Evaluate model
    predictions, targets, ious, losses = evaluate_model()
    
    print("\n=== Summary ===")
    print("✓ Direct regression approach (no training)")
    print("✓ Uses convolution features to directly compute bounding box")
    print("✓ Fast inference - no neural network weights to optimize")
    print(f"✓ Mean IoU: {np.mean(ious):.3f}")
    print(f"✓ MSE: {np.mean(losses):.6f}")

if __name__ == "__main__":
    main()
