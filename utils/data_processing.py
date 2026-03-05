import cv2
import os
import numpy as np
import random
from pathlib import Path

def augment_image(image):
    """Menerapkan kombinasi augmentasi akurat ke satu gambar."""
    aug_img = image.copy()
    h, w = aug_img.shape[:2]
    
    # 1. Random Rotation (-90 to 90 degrees) with safe border reflection
    if random.random() > 0.3:
        angle = random.uniform(-90, 90)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 2. Random Scale / Zoom (80% to 120%)
    if random.random() > 0.4:
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 3. Flips
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1) # Horizontal
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 0) # Vertical
        
    # 4. Color Jittering (Brightness, Contrast, Saturation) via HSV
    if random.random() > 0.5:
        # Pindahkan ke HSV untuk mengatur hue/saturation secara akurat
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Penyesuaian Hue (-10 to 10) - cukup tipis agar tidak mengubah tipe penyakit
        hsv[:, :, 0] = hsv[:, :, 0] + random.uniform(-10, 10)
        
        # Penyesuaian Saturation (0.8 to 1.2)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
        
        # Penyesuaian Value/Brightness (0.8 to 1.2)
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.8, 1.2)
        
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # 5. Edge-Preserving Noise Reduction (Bilateral)
    if random.random() > 0.8:
        aug_img = cv2.bilateralFilter(aug_img, 9, 75, 75)
        
    return aug_img

def smart_crop(image):
    """
    Cropping akurat berbasis Edge/Contour deteksi untuk mengisolasi lesi.
    """
    # 1. Ubah ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplikasikan GaussianBlur untuk mengurangi noise sebelum thresholding
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 3. Gunakan Otsu's Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    H, W = image.shape[:2]
    
    if contours:
        # Pilih kontur dengan area terbesar
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Batasi agar bounding box logis (menghindari background aneh)
        area_ratio = (w * h) / (H * W)
        if 0.1 < area_ratio < 0.95:
            # Berikan padding / margin 10%
            margin_x = int(0.1 * w)
            margin_y = int(0.1 * h)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(W, x + w + margin_x)
            y2 = min(H, y + h + margin_y)
            
            return image[y1:y2, x1:x2]
            
    # Fallback: Central crop 80% (jika kontur gagal/citra sulit ditebak)
    return image[int(H*0.1):int(H*0.9), int(W*0.1):int(W*0.9)]

def remove_hair(image):
    """
    Algoritma DullRazor sederhana untuk menghilangkan rambut (noise linear di kulit).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Morphological BlackHat menonjolkan strukur gelap & tipis (rambut)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. Thresholding untuk membuat mask area rambut
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 3. Inpainting: isi area masked dari pixel sekitar
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted

def preprocess_image(image, target_size=(224, 224)):
    """Pre-processing sangat akurat: Smart Crop, Hair Removal, Bilateral Filter, CLAHE, Resize."""
    
    # 1. Menghilangkan Rambut (Sangat penting agar rambut tidak dianggap fitur)
    hairless_img = remove_hair(image)
    
    # 2. Smart Crop (Isolasi area lesi)
    crop_img = smart_crop(hairless_img)

    # 3. Edge-Preserving Filtering (Bilateral Filter mengalahkan Gaussian)
    # Sangat baik membersihkan kulit tanpa mengaburkan detail penyakit
    filtered_img = cv2.bilateralFilter(crop_img, 9, 75, 75)

    # 4. CLAHE (Contrast Enhancement)
    lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Clip limit rendah (1.5) agar warna tidak over-saturated
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # 5. Resize
    final_img = cv2.resize(enhanced_img, target_size)
    
    # 6. BGR to RGB (TensorFlow/Keras Native Standard)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    
    return final_img

def load_and_preprocess_dataset(data_dir, target_size=(224, 224)):
    """Membaca folder secara rekursif, melakukan preprocess, dan mengembalikan array numpy."""
    data_dir = Path(data_dir)
    images = []
    labels = []
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"Error: Folder '{data_dir.absolute()}' tidak ditemukan!")
        return np.array([]), np.array([]), []

    if not data_dir.is_dir():
        print(f"Error: '{data_dir.absolute()}' bukan sebuah folder!")
        return np.array([]), np.array([]), []

    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    for i, class_name in enumerate(class_names):
        class_path = data_dir / class_name
        for img_path in class_path.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = preprocess_image(img, target_size)
                    images.append(img)
                    labels.append(i)
                    
    return np.array(images), np.array(labels), class_names
