import cv2
import os
import numpy as np
from pathlib import Path
from utils.noise_reduction import reduce_noise

def main():
    print("==============================================")
    print("     Test Noise Reduction Image              ")
    print("==============================================")

    base_dir = Path(__file__).parent.resolve()
    # Cari gambar sampel
    sample_img_path = base_dir / "data" / "raw" / "athlete_foot" / "athlete_1.png"
    
    if not sample_img_path.exists():
        print(f"Error: Gambar sampel tidak ditemukan di {sample_img_path}")
        return

    # Load Image
    image = cv2.imread(str(sample_img_path))
    if image is None:
        print("Gagal membaca gambar.")
        return

    # Buat folder output untuk hasil test jika belum ada
    output_dir = base_dir / "output" / "test_denoised"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Memproses gambar: {sample_img_path.name}")

    # 1. Original (Simpan copy)
    cv2.imwrite(str(output_dir / "0_original.png"), image)

    # 2. Gaussian Denoising
    print("- Mencoba Gaussian Denoising...")
    img_gaussian = reduce_noise(image, method='gaussian')
    cv2.imwrite(str(output_dir / "1_gaussian.png"), img_gaussian)

    # 3. Median Denoising
    print("- Mencoba Median Denoising...")
    img_median = reduce_noise(image, method='median')
    cv2.imwrite(str(output_dir / "2_median.png"), img_median)

    # 4. Fast NLM Denoising (Rekomendasi)
    print("- Mencoba Fast Non-Local Means Denoising (NLM)...")
    img_nlm = reduce_noise(image, method='nlm', h=10)
    cv2.imwrite(str(output_dir / "3_nlm.png"), img_nlm)

    print("\n==============================================")
    print(f"Selesai! Hasil perbandingan disimpan di:\n{output_dir.absolute()}")
    print("Silakan bandingkan hasilnya secara visual.")
    print("==============================================")

if __name__ == "__main__":
    main()
