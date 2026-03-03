# NumPy-Based Rectangle Detection

A lightweight object detection project that implements a convolution-based rectangle detector from scratch using pure NumPy. No deep learning frameworks or OpenCV required.

## 🚀 Quick Start

### The Simple Way (Recommended)

```bash
# Quick demo with 50 images + analysis
./run demo

# Full pipeline with 500 images + analysis  
./run full

# Clean everything
./run clean
```

### Manual Steps

```bash
# 1. Generate datasets
python src/data/DataGenerator.py

# 2. Update config to use your dataset
# Edit src/config.py and set DATASET_DIR to your dataset path

# 3. Run evaluation
python -m src.evaluate_model

# 4. Analyze results
python -m src.scripts.analyze_results
```

## 📋 Available Commands

| **Command** | **What it does** |
|---|---|
| `./run demo` | Quick demo (50 images, full analysis) |
| `./run full` | Complete pipeline (500 images, analysis) |
| `./run grayscale` | Generate grayscale dataset only |
| `./run colored` | Generate colored dataset only |
| `./run evaluate --dataset colored` | Evaluate on specific dataset |
| `./run analyze` | Generate analysis visualizations |
| `./run clean` | Remove all generated files |

**Options:**
- `--num-images`: Number of images to generate (default: 500)
- `--dataset`: Dataset type for evaluation (grayscale/colored, default: colored)

## 🎯 Datasets

The project supports two dataset types:

| **Type** | **Background** | **Rectangle** | **Use Case** |
|---|---|---|---|
| **Grayscale** | Grayscale noise | Black | Simpler detection |
| **Colored** | Colored noise | Various colors | More challenging |

## 📁 Project Structure

```
DummyObjectDetection/
├── run                    # Simple pipeline runner
├── run_pipeline.py        # Pipeline script
├── src/
│   ├── config.py          # Configuration settings
│   ├── data/
│   │   └── DataGenerator.py    # Dataset generation
│   ├── models/
│   │   └── detector.py        # Rectangle detector
│   └── scripts/
│       ├── prepare_data.py     # Data preparation
│       ├── evaluate_model.py    # Model evaluation
│       └── analyze_results.py  # Results analysis
└── datasets/             # Generated datasets
```

## ⚙️ Configuration

Edit `src/config.py` to change:

```python
class Config:
    IMAGE_SIZE = 256          # Input image size
    DATASET_DIR = "datasets/rectangles_colored"  # Your dataset path
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
```

## 📊 Performance

Typical results on synthetic rectangle dataset:
- **Mean IoU**: ~0.65
- **IoU > 0.5**: ~82%
- **Inference Speed**: ~2-3 samples/second

## 🔧 How It Works

1. **Edge Detection**: Sobel operators for horizontal/vertical edges
2. **Adaptive Thresholding**: Statistics-based edge binarization
3. **Contour Detection**: Pure NumPy boundary-following algorithm
4. **Direct Regression**: Extract bounding box from largest contour

## 🧪 Testing

```bash
# Quick test
python -c "
from src.models.detector import SimpleRectangleDetector
import numpy as np

image = np.zeros((256, 256))
image[50:150, 75:175] = 255

detector = SimpleRectangleDetector()
result = detector.direct_regression(image)
print('Detection result:', result)
"
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy==1.26.4` - Core numerical operations
- `Pillow>=9.0.0,<11.0.0` - Image I/O
- `matplotlib>=3.5.0,<4.0.0` - Visualization
- `tqdm>=4.65.0,<5.0.0` - Progress bars
- `scikit-learn>=1.0.0,<2.0.0` - Metrics

---

**Built with pure NumPy - no OpenCV required!** 🎯
