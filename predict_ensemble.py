import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

def preprocess_for_ensemble(image_path, target_shapes):
    """
    Memuat dan melakukan preprocessing gambar asli ke dalam berbagai resolusi input 
    yang dibutuhkan oleh masing-masing model dalam ensemble.
    """
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        raise ValueError(f"Tidak dapat membaca gambar {image_path}")
        
    outputs = {}
    
    # Pre-processing standard yang sama dengan saat training (di data_processing.py)
    # 1. Central crop (sederhana untuk inferensi jika tidak pakai smart_crop)
    h, w = orig_img.shape[:2]
    crop_img = orig_img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    
    # 2. Gaussian/Bilateral filter & CLAHE 
    filtered_img = cv2.bilateralFilter(crop_img, 9, 75, 75)
    lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize ke masing-masing shape yang dibutuhkan model
    for shape in target_shapes:
        target_size = (shape[0], shape[1]) # (Width, Height)
        resized = cv2.resize(enhanced_img, target_size)
        # Normalisasi (0-1) dan Expand dims (tambah dimensi batch)
        norm_img = resized.astype('float32') / 255.0
        outputs[shape] = np.expand_dims(norm_img, axis=0)
        
    return outputs

def main(args):
    print("==============================================")
    print("      SKIN DISEASE ENSEMBLE PREDICTION        ")
    print("==============================================")
    
    # Daftar kelas aktual
    CLASS_NAMES = ['athlete_foot', 'melanoma', 'other']

    # 1. Validasi Input Image
    img_path = Path(args.img).resolve()
    if not img_path.exists():
        print(f"Error: Gambar {img_path} tidak ditemukan.")
        return

    # 2. Muat Semua Model Keras yang Diberikan
    models_paths = args.models
    if len(models_paths) < 2:
        print("Peringatan: Ensemble idealnya menggunakan 2 atau lebih model.")
    
    loaded_models = []
    required_shapes = set()
    
    print("Memuat arsitektur model...")
    for mp in models_paths:
        p = Path(mp).resolve()
        if not p.exists():
            print(f"  [X] Gagal memuat {p.name} (File tidak ditemukan)")
            continue
            
        print(f"  [v] Memuat {p.name}...")
        try:
            model = tf.keras.models.load_model(str(p))
            loaded_models.append(model)
            required_shapes.add(tuple(model.input_shape[1:])) # Exclude batch dimension
        except Exception as e:
            print(f"  [X] Gagal memuat {p.name}. Error: {e}")
            
    if not loaded_models:
        print("Error: Tidak ada satu pun model yang berhasil dimuat.")
        return

    # 3. Preprocessing Dinamis Berdasar Kebutuhan Shape
    # (Misal: ResNet butuh 224x224, Inception butuh 299x299)
    print("\nMelakukan preprocessing gambar...")
    try:
        preprocessed_inputs = preprocess_for_ensemble(str(img_path), required_shapes)
    except Exception as e:
        print(e)
        return

    # 4. Melakukan Inferensi (Soft-Voting)
    print("\nMenghitung probabilitas gabungan (Soft-Voting)...")
    ensemble_predictions = np.zeros(len(CLASS_NAMES))
    
    for i, model in enumerate(loaded_models):
        target_shape = tuple(model.input_shape[1:])
        inp = preprocessed_inputs[target_shape]
        
        # Prediksi probabilitas (softmax)
        pred = model.predict(inp, verbose=0)[0] 
        print(f"  - Model {i+1} menebak: {CLASS_NAMES[np.argmax(pred)]} ({np.max(pred)*100:.1f}%)")
        
        # Tambahkan ke total probabilitas
        ensemble_predictions += pred
        
    # Rata-rata probabilitas
    ensemble_predictions /= len(loaded_models)
    
    # 5. Keputusan Final
    final_class_idx = np.argmax(ensemble_predictions)
    final_confidence = ensemble_predictions[final_class_idx]
    final_class_name = CLASS_NAMES[final_class_idx].replace('_', ' ').title()

    print("\n==============================================")
    print("             HASIL PREDIKSI (ENSEMBLE)        ")
    print("==============================================")
    print(f"File   : {img_path.name}")
    print(f"Penyakit: {final_class_name}")
    print(f"Akurasi Gabungan: {final_confidence * 100:.2f}%")
    print("\nProbabilitas Agregat Tiap Kelas:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  - {name:<15}: {ensemble_predictions[idx]*100:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Soft-Voting Predictor Penyakit Kulit")
    parser.add_argument("img", type=str, help="Path absolute atau relative gambar kulit")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Daftar filepath model-model .keras yang akan digabungkan (pisahkan dengan spasi)")
    
    args = parser.parse_args()
    main(args)
