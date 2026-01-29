# HGA-UNET: Consensus Attention-based Residual U-Net

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Architecture](https://img.shields.io/badge/Architecture-Siamese--UNet-green)

A specialized deep learning system for high-precision image segmentation, featuring **Hybrid Global Attention** and **Parallel Siamese Encoding**.

## 📖 Overview
HGA-UNET is designed for scientific and high-resolution imaging where traditional U-Nets struggle with low contrast or complex textures. It uses a unique "Consensus" mechanism where the model generates a "fake mask" and uses its features to guide the encoder-decoder skip connections.

### Key Features
- **Siamese Parallelization**: Shared weights between image and mask encoders for 33% fewer parameters.
- **Two-Stage Attention**: Combines Local Sliding-Window and Global Cross-Window attention.
- **Weighted Tiling**: Seamless processing of large images using Hanning window stitching.
- **Statistical Analytics**: Built-in calculation of FG Mean, STD, SEM, and spatial density heatmaps.

---

## 🛠️ Installation

### 1. Prerequisites
- macOS 12.3+ (for MPS acceleration)
- Python 3.11

### 2. Setup Environment
```bash
# Clone or navigate to the project folder
cd HGA-UNET

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio pandas scikit-image opencv-python-headless gradio
```

---

## 🚀 Usage

### Interactive Terminal (Recommended)
Use this for research and fine-tuning. It prompts for noise-removal parameters at runtime.
```bash
python inference_terminal.py
```

### Detailed Browser Report
Use this for a comprehensive visual layout of the segmentation results.
```bash
# This script is used by Antigravity to generate reports
python inference_runner.py
```

---

## 📂 Project Structure
- `Model.py`: Core architecture with Siamese optimizations.
- `segmentation_pipeline.py`: Main processing and statistical engine.
- `inference_terminal.py`: Interactive user CLI.
- `PROJECT_OVERVIEW.md`: Technical brief for AI architecture review.
- `Training/`: Ground truth dataset and sample images.
- `model_best.pth`: Current best trained parameters.

---

## 🔬 Optimization Notes
The model has been optimized for **Metal Performance Shaders (MPS)**. When running on Apple Silicon, the pipeline automatically leverages the GPU for ~22% faster inference compared to standard CPU execution.
