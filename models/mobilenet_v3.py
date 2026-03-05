import tensorflow as tf
from tensorflow import keras

# TensorFlow's submodules sometimes confuse static analyzers like Pylance.
# Importing via the top-level keras module avoids "could not be resolved" warnings.
layers = keras.layers
models = keras.models

def build_mobilenet_v3_large(input_shape=(224, 224, 3), num_classes=1000):
    """
    Membangun arsitektur MobileNetV3 Large menggunakan Keras Applications.
    """
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet', # Menggunakan weights pretrained jika tersedia
        pooling='avg'
    )
    
    # Freeze base model jika ingin transfer learning, atau biarkan terbuka (unfrozen)
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])

    return model

def build_mobilenet_v3_small(input_shape=(224, 224, 3), num_classes=1000):
    """
    Membangun arsitektur MobileNetV3 Small.
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])

    return model

if __name__ == "__main__":
    # Test building the model
    model = build_mobilenet_v3_large(num_classes=2)
    model.summary()
