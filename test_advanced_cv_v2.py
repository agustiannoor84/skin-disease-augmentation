import cv2
import os
import numpy as np
from pathlib import Path
from utils.data_processing import remove_hair, smart_crop, augment_image

def main():
    print("==============================================")
    print("   Batch Visualisasi Advanced Computer Vision ")
    print("==============================================")

    base_dir = Path(__file__).parent.resolve()
    raw_dir = base_dir / "data" / "raw"
    
    if not raw_dir.exists():
        print(f"Error: Folder input '{raw_dir}' tidak ditemukan.")
        return

    output_base_dir = base_dir / "output" / "test_advanced_cv_v2"
    # buat folder output utama di awal agar kita dapat memeriksa izin tulis
    output_base_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Kumpulkan semua gambar dari data/raw
    all_images = [f for f in raw_dir.rglob("*") if f.suffix.lower() in valid_extensions]
    
    if not all_images:
        print(f"Tidak ada gambar sampel valid ditemukan di {raw_dir}")
        return
        
    print(f"Ditemukan {len(all_images)} gambar. Memulai proses batch...\n")

    for img_path in all_images:
        try:
            # Tentukan folder output berdasarkan class/subfolder
            rel_path = img_path.relative_to(raw_dir)
            class_name = rel_path.parent.name or "_root_"
            img_stem = img_path.stem
            
            # Buat folder khusus untuk visualisasi gambar ini
            img_output_dir = output_base_dir / class_name / img_stem
            img_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Memproses: {class_name}/{img_path.name}")

            # Load Image
            image = cv2.imread(str(img_path))
            if image is None or image.size == 0:
                print(f"  -> Gagal membaca atau gambar kosong: {img_path.name}.")
                continue

            # 0. Original
            cv2.imwrite(str(img_output_dir / "0_original.png"), image)
            
            # 1. Hair Removal (DullRazor)
            hairless_img = remove_hair(image)
            if hairless_img is None or hairless_img.size == 0:
                raise ValueError("Hair removal menghasilkan gambar kosong")
            cv2.imwrite(str(img_output_dir / "1_hair_removal.png"), hairless_img)

            # 2. Smart Cropping (Berbasis Kontur Lesi)
            crop_img = smart_crop(hairless_img)
            if crop_img is None or crop_img.size == 0:
                print("  -> Smart crop gagal, menggunakan gambar asli sebagai fallback.")
                crop_img = hairless_img.copy()
            cv2.imwrite(str(img_output_dir / "2_smart_crop.png"), crop_img)

            # 3. Bilateral Filtering (Edge-Preserving)
            filtered_img = cv2.bilateralFilter(crop_img, 9, 75, 75)
            cv2.imwrite(str(img_output_dir / "3_bilateral_filter.png"), filtered_img)
            
            # 4. CLAHE (Contrast Enhancement)
            lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(img_output_dir / "4_clahe_enhancement.png"), enhanced_img)

            # 5. Advanced Augmentation (3 variasi HSV, Rotasi Aman, Zoom)
            for i in range(3):
                aug_img = augment_image(enhanced_img) 
                if aug_img is None or aug_img.size == 0:
                    raise ValueError("Augmentation menghasilkan gambar kosong")
                cv2.imwrite(str(img_output_dir / f"5_augmentation_var{i+1}.png"), aug_img)

        except Exception as e:
            print(f"  !! Error memproses {img_path.name}: {e}")
            # lanjut ke file berikutnya tanpa menghentikan seluruh batch
            continue

    print("\n==============================================")
    print(f"Batch Processing Selesai!")
    print(f"Hasil visualisasi untuk semua gambar telah disimpan di:\n{output_base_dir.absolute()}")
    print("==============================================")

if __name__ == "__main__":
    main()
