import os
import warnings
import argparse

# Suppress TensorFlow dtype warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils.data_processing import load_and_preprocess_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import fungsionalitas model yang tersedia
from models.squeezenet import build_squeezenet
from models.mobilenet_v3 import build_mobilenet_v3_large
from models.inception_v3 import build_inception_v3
from models.resnet50v2 import build_resnet50_v2
from models.efficientnetv2 import build_efficientnetv2_s
from models.densenet import build_densenet121, build_densenet169, build_densenet201, build_densenet264
from models.convnext_v2 import build_convnext_v2_tiny, build_convnext_v2_small, build_convnext_v2_base, build_convnext_v2_large, build_convnext_v2_huge

def main(args):
    # 1. Tentukan Input Shape & Fungsi Pembangun Berdasarkan Pilihan Model
    model_type = args.model_type.lower()
    if model_type == 'squeezenet':
        input_shape = (224, 224, 3)
        build_fn = build_squeezenet
    elif model_type == 'mobilenetv3':
        input_shape = (224, 224, 3)
        # Akan memuat 'large' secara default (bisa dimodif)
        build_fn = build_mobilenet_v3_large 
    elif model_type == 'inceptionv3':
        input_shape = (299, 299, 3) # InceptionV3 butuh resolusi lebih besar
        build_fn = build_inception_v3
    elif model_type == 'resnet50v2':
        input_shape = (224, 224, 3)
        build_fn = build_resnet50_v2
    elif model_type == 'efficientnetv2':
        input_shape = (224, 224, 3)
        build_fn = build_efficientnetv2_s
    elif model_type.startswith('densenet'):
        input_shape = (224, 224, 3)
        # select variant based on suffix
        if model_type == 'densenet121':
            build_fn = build_densenet121
        elif model_type == 'densenet169':
            build_fn = build_densenet169
        elif model_type == 'densenet201':
            build_fn = build_densenet201
        elif model_type == 'densenet264':
            build_fn = build_densenet264
        else:
            print(f"Error: DenseNet variant '{model_type}' tidak dikenali.")
            return
    elif model_type.startswith('convnextv2'):
        input_shape = (224, 224, 3)
        # select variant based on suffix
        if model_type == 'convnextv2_tiny':
            build_fn = build_convnext_v2_tiny
        elif model_type == 'convnextv2_small':
            build_fn = build_convnext_v2_small
        elif model_type == 'convnextv2_base':
            build_fn = build_convnext_v2_base
        elif model_type == 'convnextv2_large':
            build_fn = build_convnext_v2_large
        elif model_type == 'convnextv2_huge':
            build_fn = build_convnext_v2_huge
        else:
            print(f"Error: ConvNeXt V2 variant '{model_type}' tidak dikenali.")
            return
    else:
        print(f"Error: Model tipe '{model_type}' tidak didukung.")
        return

    print(f"=== Menyiapkan Pelatihan untuk {model_type.upper()} ===")
    print(f"Input Shape : {input_shape}")
    
    # 2. Muat Dataset
    print(f"Memuat data dari {args.data_dir}...")
    X, y, class_names = load_and_preprocess_dataset(args.data_dir, target_size=(input_shape[0], input_shape[1]))
    
    if len(X) == 0:
        print(f"Error: Tidak ada data ditemukan di {args.data_dir}")
        return

    # Normalisasi Data (0-1)
    X = X.astype('float32') / 255.0
    
    # One-hot encoding Label
    y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    
    # Split Dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Kelas Ditemukan ({len(class_names)}): {class_names}")
    print(f"Data Training : {len(X_train)}")
    print(f"Data Validasi : {len(X_val)}")

    # 3. Bangun Arsitektur
    model = build_fn(input_shape=input_shape, num_classes=len(class_names))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Advanced Training Callbacks
    model_path = Path(args.model_out).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Simpan model terbaik SAJA (berdasarkan val_accuracy tertinggi)
        ModelCheckpoint(filepath=str(model_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        # Hentikan paksa jika selama 5 epoch berturut-turut val_loss tidak membaik
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        # Turunkan kecepatan belajar 50% jika 3 epoch berturut-turut val_loss stagnan
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # 5. Mulai Pelatihan
    print("\nMemulai Pelatihan Dinamis...")
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=args.epochs, 
              batch_size=args.batch_size,
              callbacks=callbacks)

    print(f"\nPelatihan Selesai. Model final tersimpan di: {model_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(description="Script Pelatihan Dinamis Multisistem")
    parser.add_argument("--model_type", type=str, default="resnet50v2", 
                        choices=['squeezenet', 'mobilenetv3', 'inceptionv3', 'resnet50v2', 'efficientnetv2',
                                 'densenet121','densenet169','densenet201','densenet264',
                                 'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge'],
                        help="Jenis arsitektur (squeezenet/mobilenetv3/inceptionv3/resnet50v2/efficientnetv2/densenetXXX/convnextv2_XXX)")
    parser.add_argument("--data_dir", type=str, default=str(base_dir / "data" / "processed"), help="Folder dataset yang sudah digodok")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah total penggulangan")
    parser.add_argument("--batch_size", type=int, default=32, help="Jumlah foto tiap batch")
    parser.add_argument("--model_out", type=str, default=str(base_dir / "output" / "best_model.keras"), help="Lokasi simpan Output Keras")
    
    args = parser.parse_args()
    
    # Jika pengguna membiarkan model_out default, ubah nama otomatis sesuai tipe modelnya
    if args.model_out.endswith("best_model.keras"):
        args.model_out = str(base_dir / "output" / f"{args.model_type}_best.keras")
        
    main(args)
