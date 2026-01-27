import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h, w = image.shape[:2]
    
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
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add class label if available
        if class_names:
            label = class_names[int(class_id)]
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
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
