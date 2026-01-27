# Simple Rectangle Detection with NumPy

A lightweight object detection project that implements a convolution-based rectangle detector from scratch using NumPy and OpenCV. This project demonstrates how to build a simple object detector for rectangles using edge detection and direct regression without requiring training or deep learning frameworks.

## Features

- **Pure NumPy/OpenCV Implementation**: Custom convolution kernels for edge detection
- **No Training Required**: Direct regression from convolution features
- **Minimal Dependencies**: Uses only NumPy, OpenCV, and scikit-learn
- **Edge-Based Detection**: Sobel operators for horizontal/vertical edges and Laplacian for corners
- **Real-time Performance**: Fast inference with analytical computation
- **Comprehensive Tools**: Data generation, inference, analysis, and visualization scripts

## Project Structure

```
DummyObjectDetection/
├── README.md
├── requirements.txt
├── run.py               # Optional convenience wrapper
└── src/
    ├── __init__.py
    ├── config.py        # Configuration settings
    ├── train.py         # Model evaluation script
    ├── models/          # Model implementations
    │   ├── __init__.py
    │   └── detector.py  # NumPy-based rectangle detector
    ├── data/            # Data handling utilities
    │   ├── __init__.py
    │   ├── DataGenerator.py  # Dataset generation script
    │   └── dataset.py   # Custom dataset and dataloader classes
    ├── scripts/         # Analysis and utility scripts
    │   ├── __init__.py
    │   ├── prepare_data.py   # Data preparation script
    │   ├── inference.py       # Inference script for single images
    │   └── analyze_results.py # Model analysis and visualization
    └── utils/           # Utility functions
        ├── __init__.py
        └── visualization.py   # Visualization utilities
```

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (once)

```bash
python -m src.scripts.prepare_data
```

This creates a dataset of rectangle images with bounding box labels in `datasets/rectangles/`.

### 3. Evaluate Model on the Whole Dataset

To run inference over the entire validation set and see how the detector performs end‑to‑end:

```bash
python -m src.train
```

This script iterates over the dataset, runs the detector on each image, and reports aggregate metrics.

### 4. Analyze and Visualize Results

```bash
python -m src.scripts.analyze_results
```

This generates detailed plots and visualizations of detection performance across the dataset.

### 5. (Optional) Single‑Image Inference Demo

If you want to run the detector on a single image and visualize the predicted box:

```bash
python -m src.scripts.inference --image datasets/rectangles/images/rectangle_0000.jpg --output result_0000.jpg
```

Change `rectangle_0000.jpg` to any specific image you want to inspect.

### 6. (Optional) Visualize Contours and Edge Maps

To see the intermediate edge maps, binary thresholding, dilation, and contours produced by the detector:

```bash
python -m src.scripts.visualize_contours --image datasets/rectangles/images/rectangle_0000.jpg --output contours_0000.jpg
```

This generates a 2×3 subplot visualization:
- Original image
- Sobel edge magnitude
- Binary edges after adaptive threshold
- Dilated edges
- All detected contours
- Final selected contour with bounding box

Use this to debug and understand how the convolution kernels produce contours at each step.

### (Optional) Use the simple `run.py` wrapper

If you prefer shorter commands, you can also use:

```bash
python run.py prepare
python run.py train
python run.py analyze
python run.py inference --image datasets/rectangles/images/rectangle_0000.jpg --output result_0000.jpg
python run.py visualize --image datasets/rectangles/images/rectangle_0000.jpg --output contours_0000.jpg
```

This generates detailed performance metrics and visualizations.

## Detection Algorithm

The `SimpleRectangleDetector` is a **fixed, analytical pipeline**. It takes an image, applies a series of custom convolutions and simple geometric reasoning, and returns a single bounding box.

At a high level, the flow is:

```text
Input image (RGB or grayscale)
        │
        ▼
Normalization (optional: [-1,1] → [0,1])
        │
        ▼
Sobel convolutions (horizontal + vertical)
        │
        ▼
Edge magnitude map √(Hx² + Hy²)
        │
        ▼
Adaptive threshold → binary edge map
        │
        ▼
Morphological dilation (3×3 conv)
        │
        ▼
Contour detection
        │
        └─► **If valid contour found** → bounding rect → normalized box
```

The final output is a 5‑element NumPy array:

```text
[confidence, x_center, y_center, width, height]
```

All coordinates are **normalized to [0, 1]** relative to image width/height.

### 1. Edge Detection Kernels

When the model is created, it initializes three 3×3 kernels:

- **Horizontal kernel** (`h_kernel`, Sobel X)
  - Highlights **horizontal edges** (top and bottom rectangle borders).
- **Vertical kernel** (`v_kernel`, Sobel Y)
  - Highlights **vertical edges** (left and right rectangle borders).
- **Corner kernel** (`c_kernel`, Laplacian)
  - General corner/edge emphasis (currently not used directly in `direct_regression`, but available).

Conceptually, for a small patch of the image:

```text
3×3 image patch      3×3 Sobel kernel
      ⊗  (element‑wise multiply & sum)  → edge response (single number)
```

Sliding this kernel across the image via `conv2d` produces **feature maps** of edge strengths.

### 2. Custom Convolution (`conv2d`)

`conv2d` is a hand‑written 2D convolution:

```text
Input image (H×W×C) ──► convert to grayscale (H×W)
Kernel (kh×kw)      ──► (or averaged if given as 3D)
        │
        ▼
Slide kernel over image with valid padding
        │
        ▼
Output feature map of size (H−kh+1) × (W−kw+1)
```

This is used for:

- Horizontal edge detection (`h_kernel`)
- Vertical edge detection (`v_kernel`)
- Morphological operations (3×3 and 5×5 all‑ones kernels)

### 3. Edge‑Based Bounding Box (Main Path)

`direct_regression` first tries to detect the box **purely from edges**:

```text
1. Normalize (if needed)
   image in [-1,1] → (image + 1)/2 in [0,1]

2. Convert to grayscale
   RGB image → mean over channels

3. Compute Sobel edges
   Hx = conv2d(gray, h_kernel)
   Hy = conv2d(gray, v_kernel)

4. Combine into edge magnitude
   edges = sqrt(Hx² + Hy²)

5. Adaptive threshold
   threshold = mean(edges) + 2·std(edges)
   edge_map = 1 if edges > threshold else 0

6. Morphological dilation
   edge_map ──conv2d with 3×3 ones──► dilated
   dilated > 0 → 1 (thickens/bridges edges)

7. Contour detection
   find_contours(dilated) → list of contours

8. Filter contours
   - Ignore contours covering ≥ 50% of the image (likely border)
   - Ignore very small areas (< 500 px)

9. Select largest valid contour
   max(contours, key=contour_area) → best_contour

10. Bounding box
    bounding_rect(best_contour) → (x, y, w, h)

11. Normalize to [0,1]
    x_center = (x + w/2) / image_width
    y_center = (y + h/2) / image_height
    width = w / image_width
    height = h / image_height

12. Return [confidence, x_center, y_center, width, height]
```

Diagrammatically:

```text
edges (float map)
   │
   ▼
thresholding → binary edges
   │
   ▼
dilation → thicker edges
   │
   ▼
contours → largest valid rectangle
   │
   ▼
axis‑aligned bounding box → normalized coordinates → output vector
```

If no suitable contour is found in this path, the detector returns:

```text
[0.0, 0.0, 0.0, 0.0, 0.0]   # confidence 0.0, no box
```

### 4. No Training Required

Unlike traditional deep learning approaches, this detector:

- **Uses fixed convolution kernels** (Sobel, Laplacian, and all‑ones for morphology).
- **Performs only analytical operations** (convolutions, thresholds, contour geometry).
- **Does not maintain any learnable parameters** and therefore has no training loop.

The `forward` method is simply:

```python
def forward(self, image):
    """Forward pass - just direct regression"""
    return self.direct_regression(image)
```

## Key Implementation Details

### Custom Convolution
The detector implements 2D convolution from scratch using NumPy, handling both 2D and 3D inputs and kernels (by averaging over channels for 3D):

```python
def conv2d(self, image, kernel):
    """2D convolution with valid padding"""
    # Handle both 2D and 3D images
    if len(image.shape) == 2:
        h, w = image.shape
        image_2d = image
    else:
        h, w, c = image.shape
        image_2d = np.mean(image, axis=2)

    # Handle both 2D and 3D kernels
    if len(kernel.shape) == 2:
        kh, kw = kernel.shape
        kernel_2d = kernel
    else:
        kh, kw, kc = kernel.shape
        kernel_2d = np.mean(kernel, axis=2)

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image_2d[i:i+kh, j:j+kw] * kernel_2d)

    return output
```

### Edge Peak Detection
The detector finds peaks in edge responses to locate rectangle boundaries:

```python
def _find_peaks(self, features):
    """Find peaks in edge feature maps"""
    # Project onto axes and find local maxima
    h_projection = np.sum(features, axis=0)  # Horizontal projection
    v_projection = np.sum(features, axis=1)  # Vertical projection
    
    # Find peak positions
    h_peaks = self._find_local_maxima(h_projection)
    v_peaks = self._find_local_maxima(v_projection)
    
    return h_peaks, v_peaks
```

### Dataset Handling
Custom dataset and dataloader classes using scikit-learn for splitting:

```python
from sklearn.model_selection import train_test_split

# Split dataset
train_indices, val_indices = train_test_split(
    indices, test_size=val_split, random_state=random_state
)
```

## Performance

The detector achieves excellent performance for rectangle detection:

- **Real-time Inference**: Direct computation without training delays
- **High Accuracy**: Precise edge detection using Sobel and Laplacian operators  
- **Low Memory Footprint**: Pure NumPy implementation with minimal overhead
- **Robust Detection**: Works on various rectangle sizes and orientations

## Configuration

The detector behavior can be customized through `config.py`:

```python
class Config:
    IMAGE_SIZE = 256          # Input image size
    NUM_CLASSES = 1           # Only detecting rectangles
    BATCH_SIZE = 32          # Processing batch size
    LEARNING_RATE = 1e-4     # (Not used - no training)
    NUM_EPOCHS = 20          # (Not used - no training)
    VAL_SPLIT = 0.2          # Validation split ratio
```

## Future Improvements

- Add support for multiple object types (circles, triangles)
- Implement adaptive thresholding for edge detection
- Add rotation-invariant detection
- Support for overlapping rectangles
- Integration with more complex computer vision pipelines

## License

This project is for educational purposes to demonstrate NumPy-based computer vision implementations and edge detection techniques.
