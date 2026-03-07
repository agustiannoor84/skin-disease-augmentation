import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Use tf.keras submodules directly so the language server can resolve them
layers = tf.keras.layers
models = tf.keras.models

# DenseNet implementation adapted from "Densely Connected Convolutional Networks"
# https://arxiv.org/abs/1608.06993
# growth_rate is number of filters added per dense layer


def dense_layer(x, growth_rate, name=None):
    """Single dense (composite) layer: BN-ReLU-1x1 Conv-BN-ReLU-3x3 Conv"""
    bn_axis = -1
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn1')(x)
    x1 = layers.Activation('relu', name=name + '_relu1')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_conv1')(x1)

    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn2')(x1)
    x1 = layers.Activation('relu', name=name + '_relu2')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_conv2')(x1)

    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_layer(x, reduction, name=None):
    """Transition layer: BN-ReLU-1x1 Conv-AveragePool
       reduction: compression factor (0 < r <= 1)
    """
    bn_axis = -1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    filters = int(tf.keras.backend.int_shape(x)[bn_axis] * reduction)
    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def dense_block(x, blocks, growth_rate, name=None):
    """A dense block with the specified number of layers (blocks)."""
    for i in range(blocks):
        x = dense_layer(x, growth_rate, name=name + '_layer' + str(i + 1))
    return x


def build_densenet(blocks, input_shape=(224, 224, 3), num_classes=1000, growth_rate=32, reduction=0.5):
    """Construct DenseNet with given configuration.

    Args:
        blocks: list of 4 integers, number of layers in each dense block
        input_shape: input image shape
        num_classes: number of output categories
        growth_rate: filters per dense layer
        reduction: compression factor at transition layers
    """
    bn_axis = -1
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1_conv')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    # Dense blocks with transition layers
    for i, num_layers in enumerate(blocks):
        x = dense_block(x, num_layers, growth_rate, name='conv' + str(i + 2))
        if i != len(blocks) - 1:
            x = transition_layer(x, reduction, name='pool' + str(i + 2))

    # Final BN+ReLU
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fc')(x)

    model = models.Model(inputs, outputs, name='DenseNet')
    return model


# convenience builders

def build_densenet121(input_shape=(224, 224, 3), num_classes=1000):
    return build_densenet([6, 12, 24, 16], input_shape, num_classes)


def build_densenet169(input_shape=(224, 224, 3), num_classes=1000):
    return build_densenet([6, 12, 32, 32], input_shape, num_classes)


def build_densenet201(input_shape=(224, 224, 3), num_classes=1000):
    return build_densenet([6, 12, 48, 32], input_shape, num_classes)


def build_densenet264(input_shape=(224, 224, 3), num_classes=1000):
    return build_densenet([6, 12, 64, 48], input_shape, num_classes)


if __name__ == '__main__':
    m = build_densenet121(num_classes=3)
    m.summary()
