import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))

from models.mobilenet_v3 import build_mobilenet_v3_large

def test_build():
    print("Mencoba membangun MobileNetV3 Large...")
    try:
        model = build_mobilenet_v3_large(input_shape=(224, 224, 3), num_classes=3)
        print("Model berhasil dibangun!")
        
        # Test Inference dengan dummy data
        dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
        preds = model.predict(dummy_input, verbose=0)
        
        print(f"Bentuk output prediksi: {preds.shape}")
        if preds.shape == (1, 3):
            print("Verifikasi Berhasil: Output sesuai dengan jumlah kelas.")
        else:
            print(f"Verifikasi Gagal: Bentuk output {preds.shape} tidak sesuai.")
            
    except Exception as e:
        print(f"Gagal membangun atau mengetes model: {e}")

if __name__ == "__main__":
    test_build()
