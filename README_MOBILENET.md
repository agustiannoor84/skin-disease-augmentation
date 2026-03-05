# MobileNetV3 for Skin Disease Classification

Project ini menggunakan arsitektur **MobileNetV3 Large** untuk melakukan klasifikasi gambar penyakit kulit. Arsitektur ini dipilih karena efisiensinya yang tinggi, cocok untuk dijalankan di perangkat mobile atau edge computing.

## Struktur Project

- `models/mobilenet_v3.py`: Definisi arsitektur model menggunakan Keras Applications.
- `train_mobilenet.py`: Script untuk melatih model pada dataset yang sudah diproses.
- `predict_mobilenet.py`: Script untuk melakukan prediksi pada satu gambar.
- `utils/data_processing.py`: Library untuk augmentasi dan preprocessing gambar.

## Persyaratan (Requirements)

Pastikan dependensi berikut sudah terinstal:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## Cara Penggunaan

### 1. Persiapan Dataset
Dataset harus diletakkan di folder `data/processed/` dengan struktur per-kelas:
```
data/processed/
├── acne/
│   ├── img1.jpg
│   └── img2.jpg
└── dermatitis/
    ├── img3.jpg
    └── img4.jpg
```

### 2. Pelatihan (Training)
Jalankan script `train_mobilenet.py` untuk melatih model:
```bash
python train_mobilenet.py --data_dir data/processed --epochs 20 --batch_size 32
```
Model akan disimpan secara otomatis ke `output/skin_disease_mobilenetv3.keras`.

### 3. Prediksi (Inference)
Untuk melakukan prediksi pada gambar baru:
```bash
python predict_mobilenet.py --image path/to/your/image.jpg
```

## Fitur Utama

- **Transfer Learning**: Menggunakan weights 'imagenet' sebagai titik awal.
- **Data Augmentation**: Dilengkapi dengan augmentasi (rotasi, flip, brightness) untuk meningkatkan akurasi.
- **Model Checkpointing**: Hanya menyimpan model dengan akurasi validasi terbaik.
