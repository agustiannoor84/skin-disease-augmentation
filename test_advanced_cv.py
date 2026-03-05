import cv2
import numpy as np
from pathlib import Path
from utils.data_processing import remove_hair, smart_crop, augment_image

def main():
    print("==============================================")
    print("   Visualisasi Advanced Computer Vision       ")
    print("==============================================")

    base_dir = Path(__file__).parent.resolve()
    
    # Cari gambar sampel
    sample_img_path = base_dir / "data" / "raw" / "melanoma" / "melanoma_1.png"
    if not sample_img_path.exists():
        # Fallback ke athlete_foot jika melanoma tidak ada
        sample_img_path = base_dir / "data" / "raw" / "athlete_foot" / "athlete_1.png"
        
    if not sample_img_path.exists():
        print(f"Error: Tidak dapat menemukan gambar sampel di folder data/raw/")
        return

    # Load Image
    image = cv2.imread(str(sample_img_path))
    if image is None:
        print("Gagal membaca gambar.")
        return

    # Buat folder output
    output_dir = base_dir / "output" / "test_advanced_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Memproses gambar: {sample_img_path.name}")

    # 0. Original
    cv2.imwrite(str(output_dir / "0_original.png"), image)
    
    # 1. Hair Removal (DullRazor)
    print("- Step 1: Menerapkan Hair Removal (DullRazor)...")
    hairless_img = remove_hair(image)
    cv2.imwrite(str(output_dir / "1_hair_removal.png"), hairless_img)

    # 2. Smart Cropping (Berbasis Kontur Lesi)
    # Kita panggil smart crop dari hasil hair removal
    print("- Step 2: Menerapkan Smart Target Cropping...")
    crop_img = smart_crop(hairless_img)
    cv2.imwrite(str(output_dir / "2_smart_crop.png"), crop_img)

    # 3. Bilateral Filtering (Edge-Preserving)
    print("- Step 3: Menerapkan Bilateral Filtering...")
    filtered_img = cv2.bilateralFilter(crop_img, 9, 75, 75)
    cv2.imwrite(str(output_dir / "3_bilateral_filter.png"), filtered_img)
    
    # 4. CLAHE (Contrast Enhancement)
    print("- Step 4: Menerapkan CLAHE (Kontras Cerdas)...")
    lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(output_dir / "4_clahe_enhancement.png"), enhanced_img)

    # 5. Advanced Augmentation (Rotasi Aman, HSV, Zoom)
    print("- Step 5: Menerapkan Advanced Augmentation (Contoh 3 variasi)...")
    # Karena kita ingin tes visual, formatnya tetap BGR
    # Buat 3 sampel augmentasi dari hasil filter
    for i in range(3):
        # Paksa algoritma mengaugmentasi ini
        # Kita panggil fungsi augment_image yang sudah di-update di augment.py
        aug_img = augment_image(enhanced_img) 
        cv2.imwrite(str(output_dir / f"5_augmentation_var{i+1}.png"), aug_img)

    print("\n==============================================")
    print(f"Selesai! Hasil visualisasi langkah demi langkah disimpan di:\n{output_dir.absolute()}")
    print("Silakan buka foldernya untuk melihat seberapa akurat algoritma bekerja.")
    print("==============================================")

if __name__ == "__main__":
    main()
