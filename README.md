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
| `./run grayscale` | Generate grayscale dataset (filled) |
| `./run colored` | Generate colored dataset (filled) |
| `./run grayscale_border` | Generate grayscale border dataset (border only) |
| `./run colored_border` | Generate colored border dataset (border only) |
| `./run generate_all` | Generate all four datasets |
| `./run evaluate` | Evaluate on ALL datasets (default) |
| `./run evaluate --dataset colored` | Evaluate on specific dataset only |
| `./run analyze` | Analyze ALL datasets (default) |
| `./run analyze --dataset colored` | Analyze specific dataset only |
| `./run kernel_analysis` | Run kernel behavior analysis (Sobel + Laplacian) |
| `./run clean` | Remove all generated files and analysis PNGs |

**Options:**
- `--num-images`: Number of images to generate (default: 500)
- `--dataset`: Dataset type for evaluation/analysis (grayscale, colored, grayscale_border, colored_border, default: all - processes all datasets)

## 📈 Enhanced Analysis System

The analysis system now generates **dataset-specific, timestamped PNGs** for easy comparison:

### Generated Files:
- `analysis_grayscale_predictions_YYYYMMDD_HHMMSS.png` - Visual predictions for grayscale dataset
- `analysis_grayscale_metrics_YYYYMMDD_HHMMSS.png` - Performance metrics for grayscale dataset  
- `analysis_colored_predictions_YYYYMMDD_HHMMSS.png` - Visual predictions for colored dataset
- `analysis_colored_metrics_YYYYMMDD_HHMMSS.png` - Performance metrics for colored dataset
- `analysis_grayscale_border_predictions_YYYYMMDD_HHMMSS.png` - Visual predictions for grayscale border dataset
- `analysis_grayscale_border_metrics_YYYYMMDD_HHMMSS.png` - Performance metrics for grayscale border dataset
- `analysis_colored_border_predictions_YYYYMMDD_HHMMSS.png` - Visual predictions for colored border dataset
- `analysis_colored_border_metrics_YYYYMMDD_HHMMSS.png` - Performance metrics for colored border dataset

### Complete Workflow for All 4 Datasets:

**Step 1: Generate each dataset type**

```bash
# 1. Generate ALL four dataset types (500 images each)
./run grayscale --num-images 500          # Creates: datasets/rectangles_grayscale/
./run colored --num-images 500            # Creates: datasets/rectangles_colored/
./run grayscale_border --num-images 500   # Creates: datasets/rectangles_grayscale_border/
./run colored_border --num-images 500    # Creates: datasets/rectangles_colored_border/
```

**Step 2: Evaluate and analyze each dataset**

For each dataset, you MUST specify which one to evaluate, then analyze:

```bash
# Grayscale (filled rectangles)
./run evaluate --dataset grayscale
./run analyze    # Creates: analysis_grayscale_*.png

# Colored (filled rectangles)
./run evaluate --dataset colored
./run analyze    # Creates: analysis_colored_*.png

# Grayscale Border (border only)
./run evaluate --dataset grayscale_border
./run analyze    # Creates: analysis_grayscale_border_*.png

# Colored Border (border only)
./run evaluate --dataset colored_border
./run analyze    # Creates: analysis_colored_border_*.png
```

**Step 3: Compare results**

```bash
# List all analysis files to compare performance
ls -la analysis_*.png

# View the PNG files to compare predictions and metrics side-by-side
# Each filename clearly shows which dataset it analyzed
```

**Important:** `./run analyze` always analyzes the dataset you most recently evaluated. The filename will indicate which dataset was analyzed.

## 📊 Dataset Comparison

Compare the model's performance across all four dataset types:

### Performance Summary

| **Dataset** | **Rectangle Style** | **Mean IoU** | **IoU > 0.5** | **Detection Difficulty** |
|---|---|---|---|---|
| **Grayscale** | Black filled | ~0.51 | ~67% | Harder - limited contrast |
| **Colored** | Color filled | ~0.58 | ~76% | Medium - color variety helps |
| **Grayscale Border** | Black border only | ~0.75 | ~100% | Easy - clear edges |
| **Colored Border** | Colored border only | ~0.75+ | ~100% | Easiest - color + clear edges |

### Key Findings

**Border-Only Datasets Perform Best:**
- Both border variants achieve ~100% detection rate (IoU > 0.5)
- Mean IoU of ~0.75 vs ~0.55 for filled rectangles
- Clear edge definition without interior interference

**Filled vs Border-Only:**
- **Filled rectangles**: Interior noise affects edge detection
- **Border-only**: Clean edges provide stronger signals
- **Recommendation**: Use border datasets for best results

### Complete Comparison Workflow

```bash
# 1. Generate all four datasets
./run generate_all --num-images 500

# 2. Evaluate each and collect results
echo "Grayscale:" && ./run evaluate --dataset grayscale | grep "Mean IoU"
echo "Colored:" && ./run evaluate --dataset colored | grep "Mean IoU"
echo "Grayscale Border:" && ./run evaluate --dataset grayscale_border | grep "Mean IoU"
echo "Colored Border:" && ./run evaluate --dataset colored_border | grep "Mean IoU"

# 3. Generate analysis PNGs for visual comparison
./run evaluate --dataset grayscale && ./run analyze
./run evaluate --dataset colored && ./run analyze
./run evaluate --dataset grayscale_border && ./run analyze
./run evaluate --dataset colored_border && ./run analyze

# 4. View all analysis files
ls -lh analysis_*.png
```

### Benefits:

- ✅ **No file conflicts** - Each analysis gets unique timestamped filenames
- ✅ **Easy comparison** - Clear dataset labels in filenames
- ✅ **Clean management** - `./run clean` removes all analysis files
- ✅ **Progress tracking** - Real-time progress bars during analysis

## 🎯 Datasets

The project supports four dataset types:

| **Type** | **Background** | **Rectangle** | **Use Case** |
|---|---|---|---|
| **Grayscale** | Grayscale noise | Black filled | Simple detection, filled rectangles |
| **Colored** | Colored noise | Various colors filled | Challenging detection, filled rectangles |
| **Grayscale Border** | Grayscale noise | Black border only | Edge detection focus, border-only |
| **Colored Border** | Colored noise | Colored border only | Color + edge detection, border-only |

### Key Improvements:
- ✅ **No Opacity**: Rectangles are completely solid - no background shows through
- ✅ **Border Options**: Both filled and border-only variants available
- ✅ **Clean Detection**: Noise only affects background, not rectangle area

## 📁 Project Structure

```
DummyObjectDetection/
├── run                    # Simple pipeline runner
├── run_pipeline.py        # Pipeline script
├── src/
│   ├── config.py          # Configuration settings
│   ├── data/
│   │   ├── DataGenerator.py    # Dataset generation (4 types)
│   │   └── dataset.py          # Dataset loading
│   ├── models/
│   │   └── detector.py         # Rectangle detector
│   └── scripts/
│       ├── prepare_data.py     # Data preparation
│       ├── evaluate_model.py   # Model evaluation
│       ├── analyze_results.py  # Results analysis
│       └── inference.py        # Inference script
├── datasets/             # Generated datasets (4 types)
└── saved_models/         # Model checkpoints (not used for this approach)
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



# Intermediate steps still need error reporting
# Colored in boxes versus just the outline (as 3rd and 4th dataset) (DONE)
# Is corner detection looking for a "hole" on the inside corner?
#       Turns out, yes! I'm thinking this is due to the shape of the kernel used for edge detection. 
# Do fully colored boxes make it harder for the model to detect edges? (Does the space on the inside of a box give a "second chance") (DONE)
# noise is still appearing in the filled box of the grayscale dataset
# When I ran "analyze", it seems to be analyzing the same dataset, which was not happening before. Is this due to new ./run pipeline changes?
# corner detection is either finding the inside corner or the outside corner, but not both. Can we get it to do both?
# The filled in box seems to be a separate problem because there is no noise on the inside of the box. 
# maybe take absolute value, but what would the threshold be? Will run into issues with setting threshold if not specific to corner detection kernel. 
# maybe take median of the absolute values?