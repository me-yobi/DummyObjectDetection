# NumPy-Based Rectangle Detection

A lightweight, from-scratch object detection system that locates rectangles in noisy images using only NumPy — no deep learning frameworks, no OpenCV, no pretrained weights. The detector uses classical convolution (Sobel edge detection), adaptive thresholding, and contour analysis to predict bounding boxes in YOLO format, demonstrating that traditional computer vision techniques remain effective when the problem structure is well-defined.

---

## Installation

```bash
git clone https://github.com/<your-username>/DummyObjectDetection.git
cd DummyObjectDetection
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Core numerical operations and convolution |
| `Pillow` | >=9.0.0 | Image I/O |
| `matplotlib` | >=3.5.0 | Visualization and analysis plots |
| `tqdm` | >=4.65.0 | Progress bars |
| `scikit-learn` | >=1.0.0 | Evaluation metrics (MSE, MAE) |

> **Note:** No GPU required. No OpenCV required. Runs on any machine with Python 3.8+.

---

## Quick Start

```bash
# Quick demo — generates 10 images per dataset, runs evaluation + full analysis
./run demo

# Full pipeline — 500 images per dataset, evaluation, and analysis
./run full

# Clean all generated files
./run clean
```

That's it. The `./run` script handles everything from dataset generation through analysis.

---

## How It Works

The detector follows a four-stage pipeline, each implemented from scratch in pure NumPy:

### 1. Edge Detection (Sobel Convolution)

Two 3×3 Sobel kernels are convolved with the input image to detect horizontal and vertical gradients:

```
Sobel X:              Sobel Y:
[-1  0  1]           [-1 -2 -1]
[-2  0  2]           [ 0  0  0]
[-1  0  1]           [ 1  2  1]
```

For color images, each RGB channel is processed independently and the edge responses are combined using the Euclidean norm across channels. This preserves color-dependent edge information rather than collapsing to grayscale first.

### 2. Adaptive Thresholding

Rather than using a fixed threshold, the detector computes a statistics-based threshold at `mean + 2σ` of the combined edge strength map. This adapts to the noise level in each image, producing a binary edge map that isolates strong edge responses.

### 3. Morphological Dilation and Contour Detection

A 3×3 dilation kernel is convolved with the binary edge map to close small gaps, followed by a custom boundary-following contour detection algorithm. The algorithm traces 8-connected boundary pixels using a clockwise neighbor search, collecting contour points until it returns to the starting pixel.

### 4. Bounding Box Regression

The largest valid contour (excluding those covering >50% of the image) is selected, and its axis-aligned bounding rectangle is computed. The result is converted to normalized YOLO format: `[class_id, x_center, y_center, width, height]` where all coordinates are in [0, 1].

---

## Why This Approach Works

### When Traditional CV Beats Deep Learning

Deep learning models excel at general-purpose object detection across diverse categories, but they come at a cost: large labeled datasets, GPU hardware, long training times, and opaque decision-making. This project demonstrates that when the detection target has a known, regular structure — rectangles with strong edge boundaries — a deterministic pipeline of convolution, thresholding, and contour analysis can achieve competitive results with none of those costs.

**Key advantages of this approach:**

- **Zero training time** — The detector is ready to use immediately. No epochs, no hyperparameter tuning, no risk of overfitting.
- **Fully interpretable** — Every step (edge map → threshold → contour → bounding box) can be inspected and understood. You can explain exactly why a detection succeeded or failed.
- **Minimal dependencies** — Only NumPy is needed for the core detection logic. No framework lock-in, no version conflicts, no GPU requirements.
- **Deterministic results** — Same input always produces the same output. No stochastic weight initialization or data augmentation variance.

### Real-World Applications

This type of approach is practical in any domain where:

- **The target structure is known a priori** — e.g., detecting printed circuit board outlines, document boundaries in scanned pages, or structural elements in satellite imagery.
- **Resources are constrained** — Embedded systems, edge devices, or environments where installing PyTorch/TensorFlow is impractical.
- **Interpretability is required** — Medical imaging, industrial quality control, or safety-critical systems where every detection decision must be auditable.
- **Rapid prototyping is needed** — Validating that a detection problem is solvable before investing in a deep learning pipeline.

### What Makes This a Good Regression Use Case

Bounding box prediction is fundamentally a regression problem: the model must predict four continuous values (x, y, width, height), not just a class label. In the deep learning setting, this requires a carefully designed loss function, anchor boxes, and non-maximum suppression. Here, the "regression" is implicit — the contour's bounding rectangle directly yields the coordinates. This works because the Sobel kernels act as fixed feature extractors that reliably encode edge position, and the contour algorithm converts those features into spatial coordinates without any learned parameters.

---

## Dataset Types

The project generates four synthetic dataset variants to explore how rectangle appearance affects detection performance:

| **Dataset** | **Background** | **Rectangle Style** | **Edge Characteristics** |
|---|---|---|---|
| **Grayscale** | Grayscale noise | Black, filled | Weak edges — interior uniform with background noise |
| **Colored** | Colored noise | Various colors, filled | Moderate edges — color contrast helps but interior is uniform |
| **Grayscale Border** | Grayscale noise | Black, border only | Strong edges — two transitions per side (outer + inner) |
| **Colored Border** | Colored noise | Colored border only | Strongest edges — color contrast + double edge transitions |

Each image is 256×256 pixels with a randomly sized and positioned rectangle (32–128 px). Additive noise is applied only to the background region, keeping the rectangle area clean.

### Performance Comparison

| **Dataset** | **Mean IoU** | **IoU > 0.5** | **Key Insight** |
|---|---|---|---|
| Grayscale (filled) | ~0.51 | ~67% | Interior uniformity weakens edge contrast |
| Colored (filled) | ~0.58 | ~76% | Color variation improves edge detectability |
| Grayscale Border | ~0.75 | ~100% | Double edge transitions produce strong Sobel responses |
| Colored Border | ~0.75+ | ~100% | Best performance — color + clean edges |

**Why border rectangles outperform filled rectangles:** A border-only rectangle produces two distinct edge transitions per side (background→border, border→interior), each generating a strong gradient response in the Sobel kernels. A filled rectangle only produces one transition per side (background→interior), and the interior's uniform color produces no gradient at all, making the rectangle harder to distinguish from background noise.

---

## Usage

### Available Commands

All commands are run through the `./run` pipeline script:

| **Command** | **Description** |
|---|---|
| `./run demo` | Quick demo with 10 images per dataset + full analysis |
| `./run full` | Complete pipeline: generate datasets, evaluate, analyze |
| `./run setup` | Generate all four datasets with default settings |
| `./run grayscale` | Generate grayscale (filled) dataset |
| `./run colored` | Generate colored (filled) dataset |
| `./run grayscale_border` | Generate grayscale border-only dataset |
| `./run colored_border` | Generate colored border-only dataset |
| `./run generate_all` | Generate all four datasets |
| `./run evaluate` | Evaluate on all datasets (default) |
| `./run evaluate --dataset colored` | Evaluate on a single dataset |
| `./run analyze` | Analyze results for all datasets (default) |
| `./run analyze --dataset colored` | Analyze a single dataset |
| `./run kernel_analysis` | Run Sobel kernel behavior analysis |
| `./run clean` | Remove all generated datasets and analysis files |

### Options

- `--num-images N` — Number of images to generate per dataset (default: 500)
- `--dataset TYPE` — Dataset to evaluate/analyze: `grayscale`, `colored`, `grayscale_border`, `colored_border`, or `all` (default: `all`)

### Typical Workflows

**Full comparison across all dataset types:**

```bash
./run generate_all --num-images 500
./run evaluate
./run analyze
./run kernel_analysis
```

**Single dataset evaluation:**

```bash
./run colored_border --num-images 200
./run evaluate --dataset colored_border
./run analyze --dataset colored_border
```

### Generated Output Files

Evaluation and analysis produce timestamped PNG files for easy comparison:

- `analysis_<dataset>_predictions_<timestamp>.png` — Visual predictions with ground truth overlay
- `analysis_<dataset>_metrics_<timestamp>.png` — IoU, center error, and size error distributions
- `kernel_behavior_analysis.png` — Sobel kernel response visualization across dataset types
- `edge_profile_analysis.png` — Cross-section intensity and edge profiles
- `quantitative_edge_analysis.png` — Statistical edge strength comparison

---

## Project Structure

```
DummyObjectDetection/
├── run                          # Bash entry point for all pipeline commands
├── run_pipeline.py              # Pipeline orchestrator (argparse-based)
├── requirements.txt             # Python dependencies
├── src/
│   ├── config.py                # Centralized configuration (image size, paths, etc.)
│   ├── evaluate_model.py        # Model evaluation with IoU, MSE, MAE metrics
│   ├── data/
│   │   ├── DataGenerator.py     # Synthetic dataset generation (4 variants)
│   │   └── dataset.py           # Dataset loading and batching utilities
│   ├── models/
│   │   └── detector.py          # Core detector: Sobel convolution, contour detection, regression
│   ├── scripts/
│   │   ├── prepare_data.py      # CLI wrapper for dataset generation
│   │   ├── analyze_results.py   # Evaluation visualization and metric distributions
│   │   ├── kernel_analysis.py   # Sobel kernel behavior analysis and edge profiles
│   │   ├── quantitative_kernel_analysis.py  # Statistical edge strength comparison
│   │   └── inference.py         # Single-image inference script
│   └── utils/
│       └── visualization.py     # Shared plotting utilities
├── datasets/                    # Generated datasets (created by ./run)
└── saved_models/                # Model save directory (unused — no trainable weights)
```

---

## Configuration

Edit `src/config.py` to modify default settings:

```python
class Config:
    IMAGE_SIZE = 256              # Input image dimensions (square)
    NUM_CLASSES = 1               # Single class (rectangle)
    BATCH_SIZE = 32               # Batch size for evaluation
    DATASET_DIR = "..."           # Active dataset path (auto-updated by ./run)
    VAL_SPLIT = 0.2               # Validation split ratio
```

The `./run` pipeline automatically updates `DATASET_DIR` when switching between datasets, so manual config editing is only needed for custom workflows.

---

## Testing

Verify the detector works on a simple synthetic image:

```python
from src.models.detector import SimpleRectangleDetector
import numpy as np

# Create a 256x256 image with a white rectangle on black background
image = np.zeros((256, 256, 3), dtype=np.uint8)
image[50:150, 75:175] = 255

detector = SimpleRectangleDetector()
result = detector.direct_regression(image)
# result = [class_id, x_center, y_center, width, height] (normalized)
print(f"Detection: class={result[0]:.0f}, box={result[1:]}")
```

---

## License

This project is available for educational use.
