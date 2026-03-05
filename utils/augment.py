import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# Tambahkan root directory ke sys.path agar bisa dijalankan mandiri
sys.path.append(str(Path(__file__).resolve().parent.parent))

def random_rotation(image):
    """Memutar gambar (-90 hingga 90 derajat) dengan aman (Border Reflect)."""
    angle = random.uniform(-90, 90)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def random_scale(image):
    """Zoom in/out (80% - 120%) yang realistis untuk lesi kulit."""
    h, w = image.shape[:2]
    scale = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

def accurate_color_jitter(image):
    """Mengubah warna (HSV) secara akurat tanpa merusak profil warna penyakit."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift tipis
    hsv[:, :, 0] = hsv[:, :, 0] + random.uniform(-10, 10)
    # Saturation
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
    # Brightness / Value
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.8, 1.2)
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def advanced_noise_reduction(image):
    """Bilateral filter yang mempertahankan detail tapi mengurangi noise minor."""
    return cv2.bilateralFilter(image, 9, 75, 75)

def augment_image(image):
    """Menerapkan kombinasi augmentasi akurat ke satu gambar."""
    aug_img = image.copy()
    
    # Probabilitas tinggi untuk operasi geometri aman
    if random.random() > 0.3:
        aug_img = random_rotation(aug_img)
    if random.random() > 0.4:
        aug_img = random_scale(aug_img)
    if random.random() > 0.5:
        aug_img = horizontal_flip(aug_img)
    if random.random() > 0.5:
        aug_img = vertical_flip(aug_img)
        
    # Probabilitas untuk color/brightness adjustments
    if random.random() > 0.5:
        aug_img = accurate_color_jitter(aug_img)
        
    # Filter tambahan
    if random.random() > 0.8:
        aug_img = advanced_noise_reduction(aug_img)
        
    return aug_img

def augment_images(input_dir, output_dir, num_augmentations_per_image=5):
    """Membaca gambar dari input_dir secara rekursif, melakukan augmentasi akurat."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Folder input '{input_path}' tidak ditemukan!")
        return

    images_found = False
    for img_file in input_path.rglob("*"):
        if img_file.suffix.lower() in valid_extensions:
            images_found = True
            relative_path = img_file.relative_to(input_path)
            target_folder = output_path / relative_path.parent
            target_folder.mkdir(parents=True, exist_ok=True)
            
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            print(f"Augmentasi {img_file.name} ...")
            
            for i in range(num_augmentations_per_image):
                aug_image = augment_image(image)
                out_filename = f"{img_file.stem}_aug_{i+1}{img_file.suffix}"
                out_filepath = target_folder / out_filename
                cv2.imwrite(str(out_filepath), aug_image)
                
    if not images_found:
        print(f"Tidak ada gambar dengan format valid ditemukan di dalam '{input_dir}'")
    else:
        print(f"\nSelesai! Gambar hasil augmentasi presisi tinggi disimpan di: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script Augmentasi Presisi Penyakit Kulit")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Folder berisi gambar original")
    parser.add_argument("--output_dir", type=str, default="data/augmented", help="Folder hasil augmentasi")
    parser.add_argument("--num_aug", type=int, default=5, help="Jumlah gambar baru per gambar asli")
    
    args = parser.parse_args()
    augment_images(args.input_dir, args.output_dir, args.num_aug)
