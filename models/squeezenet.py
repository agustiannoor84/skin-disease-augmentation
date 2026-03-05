import tensorflow as tf

# use tf.keras directly to avoid analyzer warnings
layers = tf.keras.layers
models = tf.keras.models

def fire_module(x, squeeze, expand, name):
    """
    Fire module: Squeeze layer (1x1) followed by Expand layer (1x1 and 3x3).
    """
    # Squeeze
    squeeze_x = layers.Conv2D(squeeze, (1, 1), padding='valid', activation='relu', name=name + '_squeeze')(x)

    # Expand 1x1
    expand_1x1 = layers.Conv2D(expand, (1, 1), padding='valid', activation='relu', name=name + '_expand1x1')(squeeze_x)

    # Expand 3x3
    expand_3x3 = layers.Conv2D(expand, (3, 3), padding='same', activation='relu', name=name + '_expand3x3')(squeeze_x)

    # Concatenate
    x = layers.Concatenate(axis=-1, name=name + '_concat')([expand_1x1, expand_3x3])
    return x

def build_squeezenet(input_shape=(224, 224, 3), num_classes=1000):
    """
    Membangun arsitektur SqueezeNet v1.1.
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    # Fire blocks 2-5
    x = fire_module(x, squeeze=16, expand=64, name='fire2')
    x = fire_module(x, squeeze=16, expand=64, name='fire3')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    # Fire blocks 4-5 are actually 4-5 in v1.1 but following fire2,3
    x = fire_module(x, squeeze=32, expand=128, name='fire4')
    x = fire_module(x, squeeze=32, expand=128, name='fire5')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    # Fire blocks 6-9
    x = fire_module(x, squeeze=48, expand=192, name='fire6')
    x = fire_module(x, squeeze=48, expand=192, name='fire7')
    x = fire_module(x, squeeze=64, expand=256, name='fire8')
    x = fire_module(x, squeeze=64, expand=256, name='fire9')

    # Dropout & Final Conv
    x = layers.Dropout(0.5, name='drop9')(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='valid', name='conv10')(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='avgpool10')(x)
    outputs = layers.Activation('softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='SqueezeNet_v1.1')
    return model

if __name__ == "__main__":
    model = build_squeezenet()
    model.summary()
