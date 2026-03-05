import cv2
import numpy as np

def reduce_noise(image, method='nlm', h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """
    Mengurangi noise pada citra menggunakan berbagai metode.
    
    Args:
        image: Citra input (numpy array).
        method: 'gaussian', 'median', atau 'nlm' (Fast Non-Local Means).
        h, hColor, templateWindowSize, searchWindowSize: Parameter khusus untuk NLM.
        
    Returns:
        Citra hasil denoising.
    """
    if method == 'gaussian':
        # Gaussian Blur: Bagus untuk noise Gaussian ringan
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif method == 'median':
        # Median Blur: Sangat efektif untuk salt-and-pepper noise
        return cv2.medianBlur(image, 5)
    
    elif method == 'nlm':
        # Fast Non-Local Means Denoising: 
        # Menjaga detail tekstur lebih baik daripada blur standar.
        # Jika gambar berwarna (3 channel)
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, hColor, templateWindowSize, searchWindowSize
            )
        else:
            # Jika gambar grayscale
            return cv2.fastNlMeansDenoising(
                image, None, h, templateWindowSize, searchWindowSize
            )
    
    else:
        print(f"Metode '{method}' tidak dikenali. Mengembalikan gambar asli.")
        return image

if __name__ == "__main__":
    from pathlib import Path
    
    # Ambil root direktori proyek (2 level ke atas dari /utils/noise_reduction.py)
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    
    # Cari gambar sampel apa pun di dalam data/raw/
    sample_images = list(raw_dir.glob("*/*.*"))
    
    if sample_images:
        sample_img_path = sample_images[0]
        print(f"Menggunakan gambar sampel: {sample_img_path.name}")
        
        img = cv2.imread(str(sample_img_path))
        if img is not None:
            # Gunakan algoritma NLM untuk mengurangi noise
            denoised = reduce_noise(img, method='nlm')
            
            # Buat folder untuk menyimpan hasil tes ini
            out_dir = base_dir / "output" / "test_noise_reduction"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Simpan hasil komparasi
            cv2.imwrite(str(out_dir / f"original_{sample_img_path.name}"), img)
            cv2.imwrite(str(out_dir / f"denoised_{sample_img_path.name}"), denoised)
            
            print("Fungsi reduce_noise berhasil dijalankan dengan gambar asli.")
            print(f"Hasil komparasi disimpan di:\n{out_dir}")
        else:
            print("Gagal membaca gambar secara utuh.")
    else:
        print(f"Error: Tidak ada gambar sampel ditemukan di {raw_dir}")
