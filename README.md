# NumPy-Based Rectangle Detection

A lightweight object detection project that implements a convolution-based rectangle detector from scratch using pure NumPy. This project demonstrates how to build a complete object detection pipeline without deep learning frameworks or OpenCV dependencies.

## 🎯 Project Overview

This detector uses **direct regression from convolution features** to find rectangles in images:
- **Edge Detection**: Custom Sobel operators for horizontal/vertical edges
- **Corner Detection**: Laplacian kernel for corner features  
- **Contour Analysis**: Pure NumPy boundary-following algorithm
- **No Training**: Direct analytical computation from image features

## ✨ Key Features

- **Pure NumPy Implementation**: Complete computer vision pipeline without OpenCV
- **Custom Contour Detection**: Boundary-following algorithm with shoelace formula area calculation
- **Zero Training Required**: Direct regression from convolution features
- **Minimal Dependencies**: Only NumPy, PIL, matplotlib, and scikit-learn
- **Transparent Algorithms**: Full visibility into how contours are detected and processed
- **Real-time Performance**: Fast inference with analytical computation
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## 📁 Project Structure

```
DummyObjectDetection/
├── README.md                    # This file - comprehensive documentation
├── requirements.txt              # Python dependencies (no OpenCV!)
├── run.py                      # Convenience wrapper script
└── src/
    ├── __init__.py
    ├── config.py                 # Configuration settings
    ├── evaluate_model.py          # Model evaluation script (renamed from train.py)
    ├── models/                  # Model implementations
    │   ├── __init__.py
    │   └── detector.py          # Pure NumPy rectangle detector
    ├── data/                    # Data handling utilities
    │   ├── __init__.py
    │   ├── DataGenerator.py       # Dataset generation script
    │   └── dataset.py           # Custom dataset class
    ├── scripts/                 # Analysis and utility scripts
    │   ├── __init__.py
    │   ├── prepare_data.py       # Data preparation script
    │   ├── inference.py          # Single image inference
    │   ├── analyze_results.py    # Model analysis and visualization
    │   └── visualize_contours.py # Contour visualization tools
    └── utils/                   # Utility functions
        ├── __init__.py
        └── visualization.py     # Visualization utilities
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- No OpenCV required!

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `numpy==1.26.4` - Core numerical operations
- `Pillow>=9.0.0,<11.0.0` - Image I/O operations
- `matplotlib>=3.5.0,<4.0.0` - Visualization and plotting
- `tqdm>=4.65.0,<5.0.0` - Progress bars
- `scikit-learn>=1.0.0,<2.0.0` - Metrics and data splitting

## 🏃‍♂️ Quick Start

### 1. Generate Dataset (once)

Create synthetic rectangle images with bounding box labels:

```bash
python -m src.scripts.prepare_data
```

This creates a dataset of rectangle images with labels in `datasets/rectangles/`:
- **Images**: `datasets/rectangles/images/rectangle_*.jpg`
- **Labels**: `datasets/rectangles/labels/rectangle_*.txt` (YOLO format)

### 2. Evaluate Model Performance

Run the detector on the entire dataset and analyze performance:

```bash
python -m src.evaluate_model
```

**What this does:**
- Evaluates detector on all dataset samples (no train/validation split needed)
- Reports comprehensive metrics (IoU, MSE, MAE)
- Generates visualization of sample predictions
- Saves `regression_predictions.png` with visual results

**Expected Output:**
```
=== Rectangle Detection with Direct Regression ===
No training required - direct computation from convolution features!

Total samples: 500
Evaluating on all samples...
Evaluating: 100%|████████████████████| 16/16 [00:03<00:00,  4.67it/s]

=== Evaluation Results ===
Total samples: 500
IoU (Intersection over Union):
  Mean: 0.6522
  Samples with IoU > 0.5: 408 / 500 (81.6%)
```

### 3. Detailed Analysis and Visualization

Generate comprehensive analysis plots and statistics:

```bash
python -m src.scripts.analyze_results
```

**Generates:**
- `validation_predictions.png` - Visual comparison of predictions vs ground truth
- `metrics_distribution.png` - Distribution of IoU, center error, and size error
- Detailed console output with performance statistics

### 4. Single Image Inference

Run the detector on a single image:

```bash
python -m src.scripts.inference --image datasets/rectangles/images/rectangle_0000.jpg --output result_0000.jpg
```

**Options:**
- `--image`: Path to input image
- `--output`: Path to save result with bounding box drawn

### 5. Convenience Wrapper

Quick start with the main script:

```bash
python run.py
```

This runs the full pipeline: data generation → evaluation → analysis.

## 🔧 How It Works

### 1. Edge Detection Pipeline

```python
# Horizontal edges (Sobel X)
h_edges = conv2d(image, sobel_x_kernel)

# Vertical edges (Sobel Y)  
v_edges = conv2d(image, sobel_y_kernel)

# Combined edge strength
edges = sqrt(h_edges² + v_edges²)
```

### 2. Adaptive Thresholding

```python
# Statistics-based threshold
edge_mean = mean(edges)
edge_std = std(edges)
threshold = edge_mean + 2 * edge_std

# Binary edge map
edge_map = (edges > threshold) * 255
```

### 3. Custom Contour Detection

**No OpenCV!** Pure NumPy implementation:

1. **Boundary Detection**: Find pixels with background neighbors
2. **Contour Tracing**: 8-connected boundary following algorithm
3. **Area Calculation**: Shoelace formula for polygon area
4. **Bounding Box**: Min/max coordinates of contour points

### 4. Direct Regression

```python
# Find largest valid contour
largest_contour = max(contours, key=contour_area)

# Extract bounding rectangle
x, y, w, h = bounding_rect(largest_contour)

# Normalize to [0, 1] coordinates
x_center = (x + w/2) / image_width
y_center = (y + h/2) / image_height
```

## 📊 Performance Metrics

Typical performance on synthetic rectangle dataset:

| **Metric** | **Value** |
|---|---|
| **Mean IoU** | ~0.65 |
| **IoU > 0.5** | ~82% |
| **IoU > 0.7** | ~67% |
| **MSE** | ~0.19 |
| **Inference Speed** | ~2-3 samples/second |

## 🎨 Visualization Examples

The project generates several types of visualizations:

1. **Prediction Comparison**: Ground truth vs predicted bounding boxes
2. **Contour Analysis**: Edge maps, binary maps, and detected contours
3. **Metrics Distribution**: Histograms of IoU, position error, size error
4. **Pipeline Debug**: Step-by-step visualization of detection process

## 🔍 Debugging and Analysis

### Contour Visualization

Debug the contour detection pipeline:

```bash
python -m src.scripts.visualize_contours --image path/to/image.jpg
```

This shows:
- Original image
- Edge detection results
- Binary edge map
- All detected contours
- Final selected contour with bounding box

### Custom Parameters

Modify detection parameters in `src/config.py`:

```python
class Config:
    IMAGE_SIZE = 256          # Input image size
    DATASET_DIR = "datasets/rectangles"
    # Add custom parameters as needed
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/  # If tests exist
```

Or quick manual test:

```bash
python -c "
from src.models.detector import SimpleRectangleDetector
import numpy as np

# Test with simple rectangle
image = np.zeros((256, 256))
image[50:150, 75:175] = 255

detector = SimpleRectangleDetector()
result = detector.direct_regression(image)
print('Detection result:', result)
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is provided as educational code to demonstrate computer vision concepts with NumPy.

## 🔬 Learning Objectives

This project teaches:
- **Convolution Operations**: Manual implementation of Sobel and Laplacian kernels
- **Edge Detection**: How gradient-based edge detection works
- **Contour Algorithms**: Boundary following and polygon area calculation
- **Computer Vision Pipeline**: End-to-end object detection without deep learning
- **NumPy Mastery**: Advanced array operations for image processing

## 🚀 Extensions

Potential improvements:
- **Multiple Object Detection**: Handle multiple rectangles per image
- **Shape Classification**: Distinguish rectangles from other shapes
- **Rotation Invariance**: Detect rotated rectangles
- **Real Images**: Test on natural images with rectangles
- **Performance Optimization**: Vectorized operations for speed

---

**Built with pure NumPy - no OpenCV required!** 🎯
