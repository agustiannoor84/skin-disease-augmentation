import tensorflow as tf

# Use tf.keras submodules directly so the language server can resolve them
layers = tf.keras.layers
models = tf.keras.models

# ConvNeXt V2 implementation adapted from "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
# https://arxiv.org/abs/2301.00808


class GlobalResponseNorm(layers.Layer):
    """Global Response Normalization (GRN) layer from ConvNeXt V2"""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            name="beta",
        )

    def call(self, inputs):
        # Global average pooling along spatial dimensions
        gx = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        # L2 normalization with numerical stability
        norm = tf.sqrt(tf.reduce_sum(tf.square(gx), axis=-1, keepdims=True) + self.eps)
        nx = gx / norm
        # Apply gamma and beta
        return inputs * (self.gamma * nx + self.beta)


class ConvNeXtBlock(layers.Layer):
    """ConvNeXt V2 Block"""

    def __init__(self, dim, layer_scale_init_value=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value
        
        # Create all sublayers in __init__
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same', use_bias=False)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.conv1 = layers.Conv2D(4 * dim, kernel_size=1, use_bias=False)
        self.gelu = layers.GELU()
        self.conv2 = layers.Conv2D(dim, kernel_size=1, use_bias=False)
        self.grn = GlobalResponseNorm()
        self.add = layers.Add()

    def build(self, input_shape):
        # Layer scale parameter
        if self.layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(self.layer_scale_init_value),
                trainable=True,
                dtype=tf.float32,
                name="gamma"
            )

    def call(self, inputs):
        shortcut = inputs

        # Depthwise conv
        x = self.dwconv(inputs)
        x = self.ln1(x)

        # Pointwise convs
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)

        # GRN
        x = self.grn(x)

        # Layer scale
        if self.layer_scale_init_value > 0:
            x = x * self.gamma

        # Residual connection
        x = self.add([shortcut, x])

        return x


def convnext_block(x, dim, layer_scale_init_value=1e-6, name=None):
    """ConvNeXt V2 block wrapper"""
    if name is None:
        name = f'convnext_block_dim{dim}'
    block = ConvNeXtBlock(dim, layer_scale_init_value, name=name)
    return block(x)


def convnext_downsample(x, dim, name=None):
    """Downsampling layer for ConvNeXt V2"""
    if name is None:
        name = f'downsample_dim{dim}'
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln')(x)
    x = layers.Conv2D(dim, kernel_size=2, strides=2, use_bias=False, name=f'{name}_conv')(x)
    return x


def build_convnext_v2(depths, dims, input_shape=(224, 224, 3), num_classes=1000, layer_scale_init_value=1e-6):
    """Build ConvNeXt V2 model

    Args:
        depths: list of integers, number of blocks in each stage
        dims: list of integers, feature dimensions for each stage
        input_shape: input image shape
        num_classes: number of output classes
        layer_scale_init_value: initial value for layer scale parameters
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(dims[0], kernel_size=4, strides=4, use_bias=False, name='stem_conv')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name='stem_ln')(x)

    # Stages
    for i, (depth, dim) in enumerate(zip(depths, dims)):
        for j in range(depth):
            x = convnext_block(x, dim, layer_scale_init_value, name=f'stage{i+1}_block{j+1}')

        if i < len(depths) - 1:
            x = convnext_downsample(x, dims[i+1], name=f'stage{i+1}_downsample')

    # Head
    x = layers.GlobalAveragePooling2D(name='head_gap')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name='head_ln')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='head_dense')(x)

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