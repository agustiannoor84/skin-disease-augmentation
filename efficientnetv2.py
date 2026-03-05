import tensorflow as tf
from tensorflow.keras import layers, models, backend

def conv_block(x, filters, kernel_size=3, strides=1, activation=True, name=None):
    """Blok Konvolusi standar dengan BN dan SiLU."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=name + '_conv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_bn')(x)
    if activation:
        x = layers.Activation('swish', name=name + '_activation')(x)
    return x

def squeeze_excitation(x, input_expand_filters, name=None):
    """Squeeze-and-Excitation block."""
    filters_se = max(1, int(input_expand_filters / 4))
    se = layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(x)
    se = layers.Reshape((1, 1, input_expand_filters), name=name + '_se_reshape')(se)
    se = layers.Conv2D(filters_se, 1, padding='same', activation='swish', name=name + '_se_reduce')(se)
    se = layers.Conv2D(input_expand_filters, 1, padding='same', activation='sigmoid', name=name + '_se_expand')(se)
    return layers.Multiply(name=name + '_se_excite')([x, se])

def mb_conv_block(x, input_filters, output_filters, expansion_ratio, kernel_size=3, strides=1, use_se=True, name=None):
    """Mobile Inverted Residual Bottleneck Block (MBConv)."""
    filters = input_filters * expansion_ratio
    shortcut = x

    # Expansion
    if expansion_ratio != 1:
        x = conv_block(x, filters, kernel_size=1, name=name + '_expand')

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False, name=name + '_dwconv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_dw_bn')(x)
    x = layers.Activation('swish', name=name + '_dw_activation')(x)

    # SE
    if use_se:
        x = squeeze_excitation(x, filters, name=name)

    # Output (Projection)
    x = layers.Conv2D(output_filters, 1, padding='same', use_bias=False, name=name + '_project')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_project_bn')(x)

    # Shortcut
    if strides == 1 and input_filters == output_filters:
        x = layers.Add(name=name + '_add')([shortcut, x])
    
    return x

def fused_mb_conv_block(x, input_filters, output_filters, expansion_ratio, kernel_size=3, strides=1, name=None):
    """Fused MBConv block (Expansion Conv3x3 replaces Conv1x1 + Depthwise)."""
    filters = input_filters * expansion_ratio
    shortcut = x

    # Fused Expansion + Depthwise
    if expansion_ratio != 1:
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=name + '_expand')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_expand_bn')(x)
        x = layers.Activation('swish', name=name + '_expand_activation')(x)
        
        # Project back
        x = layers.Conv2D(output_filters, 1, padding='same', use_bias=False, name=name + '_project')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_project_bn')(x)
    else:
        # Just a standard conv if expansion is 1
        x = layers.Conv2D(output_filters, kernel_size, strides=strides, padding='same', use_bias=False, name=name + '_fused')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9, name=name + '_fused_bn')(x)
        x = layers.Activation('swish', name=name + '_fused_activation')(x)

    # Shortcut
    if strides == 1 and input_filters == output_filters:
        x = layers.Add(name=name + '_add')([shortcut, x])

    return x

def build_efficientnetv2_s(input_shape=(224, 224, 3), num_classes=1000):
    """
    Membangun arsitektur EfficientNetV2-S Sederhana.
    Sesuai dengan spesifikasi paper:
    Stage 0: 3x3 Conv, 24 filters, stride 2
    Stage 1: Fused-MBConv, 24 filters, 1 layer, stride 1, expansion 1
    Stage 2: Fused-MBConv, 48 filters, 4 layers, stride 2, expansion 4
    Stage 3: Fused-MBConv, 64 filters, 4 layers, stride 2, expansion 4
    Stage 4: MBConv, 128 filters, 6 layers, stride 2, expansion 4, SE
    Stage 5: MBConv, 160 filters, 9 layers, stride 1, expansion 6, SE
    Stage 6: MBConv, 256 filters, 15 layers, stride 2, expansion 6, SE
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = conv_block(inputs, 24, kernel_size=3, strides=2, name='stem')

    # Stage 1: FusedMBConv (24, stride 1, exp 1)
    x = fused_mb_conv_block(x, 24, 24, 1, strides=1, name='stage1_block1')

    # Stage 2: FusedMBConv (48, stride 2, exp 4)
    x = fused_mb_conv_block(x, 24, 48, 4, strides=2, name='stage2_block1')
    for i in range(1, 4):
        x = fused_mb_conv_block(x, 48, 48, 4, strides=1, name=f'stage2_block{i+1}')

    # Stage 3: FusedMBConv (64, stride 2, exp 4)
    x = fused_mb_conv_block(x, 48, 64, 4, strides=2, name='stage3_block1')
    for i in range(1, 4):
        x = fused_mb_conv_block(x, 64, 64, 4, strides=1, name=f'stage3_block{i+1}')

    # Stage 4: MBConv (128, stride 2, exp 4, SE)
    x = mb_conv_block(x, 64, 128, 4, strides=2, use_se=True, name='stage4_block1')
    for i in range(1, 6):
        x = mb_conv_block(x, 128, 128, 4, strides=1, use_se=True, name=f'stage4_block{i+1}')

    # Stage 5: MBConv (160, stride 1, exp 6, SE)
    x = mb_conv_block(x, 128, 160, 6, strides=1, use_se=True, name='stage5_block1')
    for i in range(1, 9):
        x = mb_conv_block(x, 160, 160, 6, strides=1, use_se=True, name=f'stage5_block{i+1}')

    # Stage 6: MBConv (256, stride 2, exp 6, SE)
    x = mb_conv_block(x, 160, 256, 6, strides=2, use_se=True, name='stage6_block1')
    for i in range(1, 15):
        x = mb_conv_block(x, 256, 256, 6, strides=1, use_se=True, name=f'stage6_block{i+1}')

    # Top
    x = conv_block(x, 1280, kernel_size=1, name='top_conv')
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='EfficientNetV2_S_Custom')
    return model

if __name__ == '__main__':
    # Inisialisasi Model
    model = build_efficientnetv2_s(num_classes=10) # Contoh untuk 10 kelas
    
    # Tampilkan Ringkasn
    model.summary()

    # Contoh Kompilasi
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\nModel EfficientNetV2-S berhasil dibangun dan siap dilatih.")
