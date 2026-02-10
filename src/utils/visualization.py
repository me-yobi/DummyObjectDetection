import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_with_boxes(image, boxes, class_names=None, figsize=(10, 10)):
    """
    Plot image with predicted boxes
    Args:
        image: numpy array of shape (H, W, 3) or torch tensor
        boxes: list of [class_id, x, y, w, h] in normalized coordinates
    """
    # Convert torch tensor to numpy if needed
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    
    # Handle different image formats
    if len(image.shape) == 3 and image.shape[0] == 3:
        # If in CHW format, convert to HWC
        image = image.transpose(1, 2, 0)
    
    # Denormalize if needed (assuming normalized to [-1, 1] or [0, 1])
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Work with RGB image directly
    if len(image.shape) == 3:
        image = image.copy()
    
    h, w = image.shape[:2]
    
    plt.figure(figsize=figsize)
    axes = plt.gca()
    plt.imshow(image)
    plt.axis('off')
    
    for box in boxes:
        if len(box) != 5:
            continue
            
        class_id, x_center, y_center, width, height = box
        x_center, y_center = x_center * w, y_center * h
        width, height = width * w, height * h
        
        # Calculate top-left and bottom-right coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw rectangle using matplotlib patches
        color = 'green'  # Green
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
        axes.add_patch(rect)
        
        # Add class label if available
        if class_names:
            label = class_names[int(class_id)]
            axes.text(x1, y1 - 10, label, fontsize=10, color=color)
    
    plt.show()

def visualize_batch(images, targets, predictions=None, class_names=None, max_images=4):
    """
    Visualize a batch of images with ground truth and optionally predictions
    """
    batch_size = min(len(images), max_images)
    
    for i in range(batch_size):
        image = images[i]
        target = targets[i]
        
        plt.figure(figsize=(12, 6))
        
        # Ground truth
        plt.subplot(1, 2, 1)
        plot_image_with_boxes(image, [target.tolist()], class_names, figsize=(6, 6))
        plt.title("Ground Truth")
        
        # Predictions (if available)
        if predictions is not None:
            plt.subplot(1, 2, 2)
            plot_image_with_boxes(image, [predictions[i].tolist()], class_names, figsize=(6, 6))
            plt.title("Prediction")
        
        plt.tight_layout()
        plt.show()
