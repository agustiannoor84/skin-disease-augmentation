# TensorFlow Installation Guide - Python 3.14 Compatibility Issue

## 🚨 PROBLEM: TensorFlow Not Compatible with Python 3.14

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Root Cause:** Python 3.14 is too new. TensorFlow officially supports Python 3.8-3.11 only.

## ✅ SOLUTIONS

### Solution 1: Use Compatible Python Version (Recommended)

#### Option A: Install Python 3.11 (Most Stable)
```bash
# Download and install Python 3.11 from:
# https://www.python.org/downloads/release/python-3110/

# After installation, create virtual environment:
python3.11 -m venv skin_disease_env

# Activate environment:
# Windows: skin_disease_env\Scripts\activate
# Linux/Mac: source skin_disease_env/bin/activate

# Install dependencies:
pip install -r requirements_tf_compatible.txt
```

#### Option B: Use pyenv (Cross-platform)
```bash
# Install pyenv (if not installed)
# Windows: https://github.com/pyenv-win/pyenv-win
# Linux/Mac: https://github.com/pyenv/pyenv

# Install Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# Create virtual environment
python -m venv skin_disease_env
source skin_disease_env/bin/activate  # or skin_disease_env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements_tf_compatible.txt
```

### Solution 2: Use Conda/Miniconda (Alternative)

```bash
# Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

# Create environment with Python 3.11
conda create -n skin_disease python=3.11
conda activate skin_disease

# Install TensorFlow
conda install tensorflow numpy matplotlib scikit-learn opencv pillow
```

### Solution 3: Use Docker (Advanced)

```bash
# Use official TensorFlow Docker image
docker run -it --rm -v $(pwd):/app -w /app tensorflow/tensorflow:2.16.1-jupyter

# Inside container:
pip install opencv-python matplotlib scikit-learn pillow
```

## 📋 VERIFICATION

After installation, test with:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Test ConvNeXt V2
from models.convnext_v2 import build_convnext_v2_tiny
model = build_convnext_v2_tiny(num_classes=3)
print("Model built successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
```

## 🎯 RECOMMENDED SETUP

1. **Install Python 3.11**
2. **Create virtual environment:**
   ```bash
   python -m venv skin_disease_env
   skin_disease_env\Scripts\activate  # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install tensorflow numpy opencv-python matplotlib scikit-learn pillow
   ```
4. **Run training:**
   ```bash
   python train.py --model_type convnextv2_base --epochs 50 --batch_size 32
   ```

## 🔧 TROUBLESHOOTING

### If installation still fails:
- Check Python version: `python --version` (should be 3.8-3.11)
- Update pip: `python -m pip install --upgrade pip`
- Clear pip cache: `pip cache purge`
- Try CPU-only version: `pip install tensorflow-cpu`

### GPU Support (Optional):
```bash
# For CUDA support, ensure compatible versions:
pip install tensorflow[and-cuda]  # or tensorflow-gpu
```

## 📚 COMPATIBILITY MATRIX

| Python Version | TensorFlow Support | Status |
|----------------|-------------------|---------|
| 3.8           | ✅ Full support   | Stable |
| 3.9           | ✅ Full support   | Stable |
| 3.10          | ✅ Full support   | Stable |
| 3.11          | ✅ Full support   | Stable |
| 3.12          | ⚠️ Limited       | Experimental |
| 3.13          | ❌ No support     | Not available |
| 3.14          | ❌ No support     | Not available |

## 🚀 QUICK START (After Setup)

```bash
# Activate environment
skin_disease_env\Scripts\activate

# Train ConvNeXt V2 Tiny
python train.py --model_type convnextv2_tiny --epochs 10 --batch_size 32

# Train ConvNeXt V2 Base (recommended)
python train.py --model_type convnextv2_base --epochs 50 --batch_size 32

# Verify installation
python verify_convnext.py
```