import os
import warnings

# Suppress TensorFlow dtype warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Use tf.keras submodules directly so the language server can resolve them
layers = tf.keras.layers
models = tf.keras.models

# ConvNeXt V2 implementation adapted from "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
# https://arxiv.org/abs/2301.00808


def build_convnext_v2(depths, dims, input_shape=(224, 224, 3), num_classes=1000, layer_scale_init_value=1e-6):
    """Build ConvNeXt V2 model using Functional API

    Args:
        depths: list of integers, number of blocks in each stage
        dims: list of integers, feature dimensions for each stage
        input_shape: input image shape
        num_classes: number of output classes
        layer_scale_init_value: initial value for layer scale parameters
    """
    inputs = layers.Input(shape=input_shape)

    # Stem: 4x4 conv with stride 4
    x = layers.Conv2D(dims[0], kernel_size=4, strides=4, use_bias=False, name='stem_conv')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name='stem_ln')(x)

    # Stages with dense blocks
    for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
        # Dense block with multiple ConvNeXt blocks
        for block_idx in range(depth):
            shortcut = x
            
            # Depthwise separable convolution path
            x = layers.DepthwiseConv2D(
                kernel_size=7, 
                padding='same', 
                use_bias=False, 
                name=f'stage{stage_idx}_block{block_idx}_dw'
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6, name=f'stage{stage_idx}_block{block_idx}_ln1')(x)

            # Pointwise expansion
            x = layers.Conv2D(
                4 * dim, 
                kernel_size=1, 
                use_bias=False, 
                name=f'stage{stage_idx}_block{block_idx}_pw1'
            )(x)
            x = layers.GELU(name=f'stage{stage_idx}_block{block_idx}_gelu')(x)

            # Pointwise reduction
            x = layers.Conv2D(
                dim, 
                kernel_size=1, 
                use_bias=False, 
                name=f'stage{stage_idx}_block{block_idx}_pw2'
            )(x)

            # Layer scale with residual
            if layer_scale_init_value > 0:
                x = layers.Multiply(name=f'stage{stage_idx}_block{block_idx}_scale')([
                    x, 
                    layers.Lambda(lambda y: tf.ones_like(y) * layer_scale_init_value)(x)
                ])

            # Residual connection
            x = layers.Add(name=f'stage{stage_idx}_block{block_idx}_add')([shortcut, x])

        # Transition/Downsampling layer between stages
        if stage_idx < len(depths) - 1:
            x = layers.LayerNormalization(epsilon=1e-6, name=f'stage{stage_idx}_downsample_ln')(x)
            x = layers.Conv2D(
                dims[stage_idx + 1], 
                kernel_size=2, 
                strides=2, 
                use_bias=False, 
                name=f'stage{stage_idx}_downsample_conv'
            )(x)

    # Head
    x = layers.GlobalAveragePooling2D(name='head_gap')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name='head_ln')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='ConvNeXt_V2')
    return model


# Predefined ConvNeXt V2 variants

def build_convnext_v2_tiny(input_shape=(224, 224, 3), num_classes=1000):
    """ConvNeXt V2 Tiny: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]"""
    return build_convnext_v2([3, 3, 9, 3], [96, 192, 384, 768], input_shape, num_classes)


def build_convnext_v2_small(input_shape=(224, 224, 3), num_classes=1000):
    """ConvNeXt V2 Small: depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]"""
    return build_convnext_v2([3, 3, 27, 3], [96, 192, 384, 768], input_shape, num_classes)


def build_convnext_v2_base(input_shape=(224, 224, 3), num_classes=1000):
    """ConvNeXt V2 Base: depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]"""
    return build_convnext_v2([3, 3, 27, 3], [128, 256, 512, 1024], input_shape, num_classes)


def build_convnext_v2_large(input_shape=(224, 224, 3), num_classes=1000):
    """ConvNeXt V2 Large: depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]"""
    return build_convnext_v2([3, 3, 27, 3], [192, 384, 768, 1536], input_shape, num_classes)


def build_convnext_v2_huge(input_shape=(224, 224, 3), num_classes=1000):
    """ConvNeXt V2 Huge: depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816]"""
    return build_convnext_v2([3, 3, 27, 3], [352, 704, 1408, 2816], input_shape, num_classes)


if __name__ == '__main__':
    # Test the model
    m = build_convnext_v2_tiny(num_classes=3)
    m.summary()