import os
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from models.mobilenet_v3 import build_mobilenet_v3_large
from utils.data_processing import load_and_preprocess_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

def main(args):
    print(f"Memuat data dari {args.data_dir}...")
    X, y, class_names = load_and_preprocess_dataset(args.data_dir)
    
    if len(X) == 0:
        print(f"Error: Tidak ada data ditemukan di {args.data_dir}")
        print("Tip: Pastikan dataset sudah diproses di folder tersebut.")
        return

    # Normalisasi (MobileNetV3 biasanya mengharapkan range [0, 255] jika menggunakan preprocessing internal, 
    # tapi di sini kita pakai manual preprocessing yang sudah ada di utils)
    X = X.astype('float32') / 255.0
    
    # One-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    
    # Split Dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Ditemukan {len(class_names)} kelas: {class_names}")
    print(f"Data Training: {len(X_train)}, Data Validasi: {len(X_val)}")

    # Build Model
    model = build_mobilenet_v3_large(input_shape=(224, 224, 3), num_classes=len(class_names))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nMemulai Pelatihan MobileNet V3...")
    
    # Callback untuk menyimpan model terbaik
    checkpoint_path = Path(args.model).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_accuracy'
    )

    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=args.epochs, 
                        batch_size=args.batch_size,
                        callbacks=[cp_callback])

    print(f"\nModel terbaik disimpan ke {checkpoint_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(base_dir / "data" / "processed"), help="Folder dataset processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default=str(base_dir / "output" / "skin_disease_mobilenetv3.keras"), help="Output model path")
    
    args = parser.parse_args()
    main(args)
