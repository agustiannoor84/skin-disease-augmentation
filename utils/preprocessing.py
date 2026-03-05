import cv2
import os
import sys
import argparse
from pathlib import Path

# Tambahkan root directory ke sys.path agar 'utils' terbaca sebagai modul
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data_processing import smart_crop, remove_hair

def preprocess_images(input_folder='data/raw', output_folder='data/pre'):
    """Membaca folder input secara rekursif, melakukan preprocess AKURAT, dan menyimpan di output_folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Error: Folder '{input_path}' tidak ditemukan.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Mencari file secara rekursif
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in valid_extensions]
    
    if not files:
        print(f"Tidak ada gambar valid ditemukan di '{input_path}'")
        return

    print(f"Memproses {len(files)} gambar dengan algoritma canggih (DullRazor, Smart Crop, Bilateral)...")
    
    for filename in files:
        relative_path = filename.relative_to(input_path)
        target_folder = output_path / relative_path.parent
        target_folder.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(filename))
        if img is None: 
            print(f"Gagal membaca {filename.name}")
            continue
            
        print(f"Memproses {filename.name} (dari {filename.parent.name}) ...")

        # 1. Hair Removal (DullRazor)
        hairless_img = remove_hair(img)

        # 2. Smart Crop berbasis Contour Deteksi
        crop_img = smart_crop(hairless_img)

        # 3. Edge-Preserving Filtering (Bilateral)
        filtered_img = cv2.bilateralFilter(crop_img, 9, 75, 75)

        # 4. CLAHE Enhancement (Contrast)
        lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # Jangan convert BGR ke RGB di sini karena cv2.imwrite butuh BGR
        # Jangan resize statis di sini agar kualitas max tetap ada untuk augmentasi
        
        # Simpan hasil akhir
        out_filepath = target_folder / f"{filename.stem}_pre{filename.suffix}"
        cv2.imwrite(str(out_filepath), enhanced_img)

    print(f"\nSelesai! Gambar pre-processing (Sangat Akurat) disimpan di: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Pre-processing Sangat Akurat")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Folder berisi gambar original")
    parser.add_argument("--output_dir", type=str, default="data/pre", help="Folder hasil pemrosesan")
    
    args = parser.parse_args()
    preprocess_images(args.input_dir, args.output_dir)
