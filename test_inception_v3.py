import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from models.inception_v3 import build_inception_v3

def main():
    print("==============================================")
    print("   Test Inisialisasi Model InceptionV3        ")
    print("==============================================")

    # InceptionV3 merekomendasikan input 299x299
    input_shape = (299, 299, 3)
    num_classes = 3

    print(f"Membangun arsitektur dengan input_shape={input_shape} dan classes={num_classes}...")
    try:
        model = build_inception_v3(
            input_shape=input_shape,
            num_classes=num_classes,
            freeze_base=True
        )
        print("-> Pembangunan arsitektur berhasil.")
        
        print("\nMenampilkan struktur 5 layer terakhir:")
        for layer in model.layers[-5:]:
            # Beberapa layer seperti Concatenate tidak punya 1 output_shape spesifik
            try:
                out_shape = layer.output_shape
            except AttributeError:
                out_shape = "Multiple / Varies"
            print(f"- {layer.name:<25} : {out_shape}")

    except Exception as e:
        print(f"-> Gagal membangun model. Error: {e}")
        return

    # Uji coba forward pass (dummy prediction)
    print("\nMenguji 'Forward Pass' dengan tensor kosong (dummy data)...")
    # Membuat 2 batch gambar kosong berukuran 299x299x3
    dummy_input = np.random.rand(2, *input_shape).astype(np.float32)
    
    try:
        predictions = model.predict(dummy_input, verbose=0)
        print(f"-> Forward pass berhasil!")
        print(f"   Shape output prediksi: {predictions.shape}  (ekspektasi: (2, 3))")
        print(f"   Contoh output logit probabilitas model: {predictions[0]}")
    except Exception as e:
        print(f"-> Forward pass gagal. Error: {e}")
        return

    print("\n==============================================")
    print("Selesai! Model InceptionV3 siap dilatih (train.py).")
    print("==============================================")

if __name__ == "__main__":
    main()
