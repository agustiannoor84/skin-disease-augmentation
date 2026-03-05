import os
import argparse
import numpy as np
import tensorflow as tf
from utils.data_processing import preprocess_image
from pathlib import Path
import cv2

def main(args):
    # Check if model exists
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"Error: Model '{model_path}' tidak ditemukan!")
        print("Tip: Jalankan 'python train_mobilenet.py' terlebih dahulu.")
        return

    # Load Model
    print(f"Memuat model dari {model_path}...")
    model = tf.keras.models.load_model(str(model_path))
    
    # Load Image
    img_path = Path(args.image).resolve()
    if not img_path.exists():
        print(f"Error: Gambar '{img_path}' tidak ditemukan!")
        return

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Tidak bisa membaca gambar di {img_path}")
        return

    # Preprocess
    processed_img = preprocess_image(image, target_size=(224, 224))
    
    # Expand dims for batch
    input_tensor = np.expand_dims(processed_img, axis=0).astype('float32') / 255.0

    # Inference
    predictions = model.predict(input_tensor)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    print(f"\nHasil Prediksi untuk {img_path.name}:")
    print(f"------------------------------------")
    print(f"Kelas ID: {class_idx}")
    print(f"Confidence: {confidence:.2%}")
    print(f"------------------------------------")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar yang akan diprediksi")
    parser.add_argument("--model", type=str, default=str(base_dir / "output" / "skin_disease_mobilenetv3.keras"), help="Path ke model .keras")
    
    args = parser.parse_args()
    main(args)
