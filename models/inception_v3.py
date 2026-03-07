import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras  # type: ignore

# import specific components via keras to keep static analysis happy
InceptionV3 = keras.applications.InceptionV3
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dropout = keras.layers.Dropout
Model = keras.models.Model

def build_inception_v3(input_shape=(299, 299, 3), num_classes=3, freeze_base=True):
    """
    Membangun model klasifikasi berdasarkan InceptionV3 pretrained dari ImageNet.
    
    Args:
        input_shape: Dimensi input gambar (default untuk InceptionV3 adalah 299x299).
        num_classes: Jumlah kelas untuk klasifikasi (default 3: athlete_foot, melanoma, other).
        freeze_base: Jika True, bobot lapisan pre-trained InceptionV3 tidak akan diperbarui 
                     selama training (Transfer Learning phase 1).
                     
    Returns:
        tf.keras.Model yang siap untuk dicompile dan dilatih.
    """
    # 1. Muat base model InceptionV3 tanpa lapisan output utama (include_top=False)
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # 2. Bekukan/Freeze lapisan base model jika diminta
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
            
    # 3. Tambahkan lapisan klasifikasi kustom di atas base model
    x = base_model.output
    
    # GlobalAveragePooling mengubah feature map menjadi vektor 1D
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    
    # Tambahkan lapisan Dense (Fully Connected) untuk mempelajari kombinasi fitur
    x = Dense(1024, activation='relu', name='dense_1024')(x)
    
    # Tambahkan Dropout untuk mencegah overfitting
    x = Dropout(0.5, name='dropout_0.5')(x)
    
    # Lapisan output final (Softmax untuk klasifikasi multi-kelas)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # 4. Satukan base model dan lapisan baru menjadi sebuah Model Keras
    model = Model(inputs=base_model.input, outputs=predictions, name='InceptionV3_Custom')
    
    return model

if __name__ == "__main__":
    # Test pembuatan model
    print("Membangun model InceptionV3...")
    model = build_inception_v3(input_shape=(299, 299, 3), num_classes=3)
    
    print("\n--- Ringkasan Model ---")
    model.summary()
    
    print("\nModel InceptionV3 berhasil dibuat dan siap digunakan.")
