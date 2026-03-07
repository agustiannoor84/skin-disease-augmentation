import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import argparse
import tensorflow as tf
from utils.data_processing import preprocess_image
from pathlib import Path

# Pemetaan index ke nama kelas
CLASS_NAMES = ['athlete_foot', 'melanoma', 'other']

def predict(img_path, model_path):
    # Check if model exists
    model_path_obj = Path(model_path).resolve()
    if not model_path_obj.exists():
        print(f"Error: Model tidak ditemukan di {model_path_obj}")
        print("Tip: Pastikan Anda sudah menjalankan pelatihan model terlebih dahulu.")
        return

    # Load Model
    print(f"Memuat model dari {model_path_obj}...")
    try:
        model = tf.keras.models.load_model(str(model_path_obj))
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return
    
    # Load and Preprocess Image
    img_path_obj = Path(img_path).resolve()
    if not img_path_obj.exists():
        print(f"Error: Gambar tidak ditemukan di {img_path_obj}")
        return

    img = cv2.imread(str(img_path_obj))
    if img is None:
        print(f"Error: Gagal membaca gambar {img_path_obj}")
        return

    # Preprocess (ResNet50V2 biasanya menggunakan 224x224)
    processed_img = preprocess_image(img, target_size=(224, 224))
    
    # Expand dims for batch and normalize
    input_tensor = np.expand_dims(processed_img, axis=0).astype('float32') / 255.0
    
    # Prediction
    print("Melakukan inferensi...")
    preds = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Unknown ({class_idx})"
    
    print(f"\n{'='*40}")
    print(f" HASIL PREDIKSI")
    print(f"{'='*40}")
    print(f"File: {img_path_obj.name}")
    print(f"Prediksi Kelas: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(description="Prediksi Penyakit Kulit menggunakan ResNet50V2")
    # nargs='?' membuat argumen menjadi opsional di CLI
    parser.add_argument("img", type=str, nargs='?', help="Path ke file gambar (opsional, akan mencari otomatis jika kosong)")
    parser.add_argument("--model", type=str, default=str(base_dir / "output" / "skin_disease_resnet50v2.keras"), help="Path ke model .keras")
    
    args = parser.parse_args()
    
    img_path = args.img
    # Jika user tidak memasukkan gambar, coba cari otomatis di folder data
    if not img_path:
        print("Tip: Anda tidak memasukkan path gambar. Mencari gambar sampel otomatis...")
        sample_paths = list(base_dir.glob("data/raw/*/*.*"))
        if sample_paths:
            img_path = str(sample_paths[0])
            print(f"Menggunakan gambar sampel: {img_path}")
        else:
            print("\nError: Tidak ada gambar yang diberikan dan tidak ditemukan gambar sampel di folder data/raw.")
            print("Cara pakai: python predict.py <path_gambar>")
            exit(1)
            
    predict(img_path, args.model)
