# ConvNeXt V2 Implementation - Skin Disease Classification

## Overview
Complete implementation of ConvNeXt V2 architecture for skin disease classification with 5 model variants.

## Model Variants
- **Tiny**: Lightweight, fastest inference (96→192→384→768 channels)
- **Small**: Balance of speed and accuracy (96→192→384→768 channels) 
- **Base**: Recommended for production (128→256→512→1024 channels)
- **Large**: High accuracy (192→384→768→1536 channels)
- **Huge**: Maximum accuracy (352→704→1408→2816 channels)

## Key Features
✅ Pure Functional API implementation (no custom layer issues)  
✅ Depthwise separable convolutions for efficiency  
✅ LayerNormalization for training stability  
✅ Proper residual connections with layer scaling  
✅ Global average pooling + dense classification head  

## Usage

### Basic Usage
```python
from models.convnext_v2 import build_convnext_v2_tiny

# Build model
model = build_convnext_v2_tiny(num_classes=3)  # For 3 classes: athlete_foot, melanoma, other

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32)
```

### Training with train.py Script
```bash
# Train Tiny variant
python train.py --model_type convnextv2_tiny --epochs 50 --batch_size 32

# Train Small variant
python train.py --model_type convnextv2_small --epochs 50 --batch_size 32

# Train Base variant (recommended)
python train.py --model_type convnextv2_base --epochs 50 --batch_size 32

# Train Large variant
python train.py --model_type convnextv2_large --epochs 50 --batch_size 32

# Train Huge variant
python train.py --model_type convnextv2_huge --epochs 50 --batch_size 32

# Custom settings
python train.py --model_type convnextv2_base --data_dir data/processed --epochs 100 --batch_size 16 --model_out output/convnext_v2_base.keras
```

## Architecture Details

### Stage Configuration
Each variant follows this stage structure:
- **Stem**: 4×4 convolution with stride 4 to reduce spatial dimensions
- **Stage 1-4**: Dense blocks with depthwise separable convolutions
- **Downsample layers**: Between stages using LayerNorm + 2×2 Conv with stride 2
- **Head**: Global average pooling + LayerNorm + Dense classification layer

### Block Components
Each ConvNeXt block contains:
1. Depthwise 7×7 convolution + LayerNorm
2. Pointwise 1×1 expansion (4x channels) + GELU activation  
3. Pointwise 1×1 reduction back to original channels
4. Layer scale multiplication for gradual contribution
5. Residual connection to input

## Performance Characteristics

| Variant | Parameters | Input Size | Speed | Accuracy |
|---------|-----------|-----------|-------|----------|
| Tiny    | ~28.6M    | 224×224   | Fast  | Good     |
| Small   | ~50.2M    | 224×224   | Fast  | Good     |
| Base    | ~88.6M    | 224×224   | Medium| Excellent|
| Large   | ~197.8M   | 224×224   | Slow  | Excellent|
| Huge    | ~660.3M   | 224×224   | Very Slow| Best |

## Model Saving/Loading
```python
# Save model
model.save('output/convnext_v2_base.keras')

# Load model
from tensorflow.keras.models import load_model
model = load_model('output/convnext_v2_base.keras')
```

## Known Warnings
TensorFlow internal dtype warnings (float4_e2m1fn) are suppressed at the module level and do not affect model functionality. These are suppressed via:
```python
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

## Advantages over Previous Models
✓ Modern architecture with state-of-the-art accuracy  
✓ Depthwise separable convolutions reduce parameters  
✓ LayerNormalization improves training stability  
✓ Pure Functional API (compatible with all TensorFlow/Keras versions)  
✓ Flexible model size options (Tiny → Huge)  

## References
- Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
  https://arxiv.org/abs/2301.00808
- Original Implementation: https://github.com/facebookresearch/ConvNeXt

## Troubleshooting
If you encounter issues:
1. Ensure TensorFlow is installed: `pip install tensorflow`
2. Check Python version compatibility (3.8+)
3. Verify CUDA/GPU support if using GPU training
4. Check data preprocessing in `utils/data_processing.py`
