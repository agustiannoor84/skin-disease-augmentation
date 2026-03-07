import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# reference submodules via tf.keras
layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend

def bottleneck_block_v2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """
    Mengimplementasikan unit Bottleneck ResNetV2 dengan Pre-activation.
    """
    filters1, filters2, filters3 = filters
    
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    if name is None:
        name = 'block' + str(backend.get_uid('block'))

    # --- JALUR PRE-ACTIVATION (BN -> ReLU) ---
    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    # --- JALUR UTAMA ---
    x_main = layers.Conv2D(filters1, 1, strides=stride, use_bias=False, name=name + '_1_conv')(preact)
    x_main = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x_main)
    x_main = layers.Activation('relu', name=name + '_1_relu')(x_main)

    x_main = layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=name + '_2_conv')(x_main)
    x_main = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x_main)
    x_main = layers.Activation('relu', name=name + '_2_relu')(x_main)

    x_main = layers.Conv2D(filters3, 1, name=name + '_3_conv')(x_main)

    # --- JALUR SHORTCUT ---
    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(filters3, 1, strides=stride, name=name + '_short_conv')(preact)

    x_final = layers.Add(name=name + '_out')([shortcut, x_main])
    return x_final

def build_resnet50_v2(input_shape=(224, 224, 3), num_classes=1000):
    """
    Membangun arsitektur ResNet50V2 penuh.
    """
    inputs = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    # --- STEM ---
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1_conv')(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # --- STAGES ---
    x = bottleneck_block_v2(x, [64, 64, 256], stride=1, conv_shortcut=True, name='conv2_block1')
    x = bottleneck_block_v2(x, [64, 64, 256], conv_shortcut=False, name='conv2_block2')
    x = bottleneck_block_v2(x, [64, 64, 256], conv_shortcut=False, name='conv2_block3')

    x = bottleneck_block_v2(x, [128, 128, 512], stride=2, conv_shortcut=True, name='conv3_block1')
    x = bottleneck_block_v2(x, [128, 128, 512], conv_shortcut=False, name='conv3_block2')
    x = bottleneck_block_v2(x, [128, 128, 512], conv_shortcut=False, name='conv3_block3')
    x = bottleneck_block_v2(x, [128, 128, 512], conv_shortcut=False, name='conv3_block4')

    x = bottleneck_block_v2(x, [256, 256, 1024], stride=2, conv_shortcut=True, name='conv4_block1')
    for i in range(2, 7):
        x = bottleneck_block_v2(x, [256, 256, 1024], conv_shortcut=False, name=f'conv4_block{i}')

    x = bottleneck_block_v2(x, [512, 512, 2048], stride=2, conv_shortcut=True, name='conv5_block1')
    x = bottleneck_block_v2(x, [512, 512, 2048], conv_shortcut=False, name='conv5_block2')
    x = bottleneck_block_v2(x, [512, 512, 2048], conv_shortcut=False, name='conv5_block3')

    # --- CLASSIFIER HEAD ---
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='ResNet50V2_Custom')
    return model

if __name__ == "__main__":
    model = build_resnet50_v2()
    model.summary()
