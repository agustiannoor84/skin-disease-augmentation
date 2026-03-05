# ResNet50 Skin Disease Classification

Project ini mengimplementasikan klasifikasi penyakit kulit menggunakan arsitektur ResNet50V2 dengan pre-processing gambar berbasis OpenCV.

## Struktur Project
- `models/`: Definisi arsitektur model ResNet50V2.
- `utils/`: Fungsi helper untuk pre-processing dan augmentasi.
- `train.py`: Script untuk melatih model.
- `predict.py`: Script untuk melakukan prediksi pada gambar tunggal.
- `data/`: Folder untuk menyimpan dataset (raw, pre-processed, augmented).

## Cara Penggunaan

### 1. Instalasi Dependensi
Pastikan Anda memiliki Python 3.9+ dan jalankan:
```bash
pip install -r requirements.txt
```

### 2. Persiapan Data
Letakkan gambar Anda di dalam folder `data/raw` dengan struktur per kelas:
```
data/raw/
    melanoma/
        img1.jpg
        img2.jpg
    basal_cell_carcinoma/
        img3.jpg
```

### 3. Pelatihan Model
Jalankan script berikut untuk melatih model:
```bash
python train.py --epochs 20 --batch_size 16
```

### 4. Prediksi
Gunakan model yang sudah dilatih (di folder `output/`) untuk mengetes gambar baru:
```bash
python predict.py --img path/to/image.jpg
```

## Troubleshooting (TensorFlow Not Found)
Jika Anda melihat error `Could not find import of tensorflow`:
1. Pastikan Anda telah menginstal tensorflow: `pip install tensorflow`
2. Jika menggunakan VS Code, pastikan **Python Interpreter** yang dipilih adalah interpreter di mana Anda menginstal library tersebut (tekan `Ctrl+Shift+P` -> `Python: Select Interpreter`).
