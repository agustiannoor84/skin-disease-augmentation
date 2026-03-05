import sys
import os
from pathlib import Path

# Tambahkan folder 'utils' ke path agar modul bisa diimport
sys.path.append(str(Path(__file__).parent / "utils"))

from preprocessing import preprocess_images
from augment import augment_images

def main():
    print("==============================================")
    print("   Skin Disease Image Processing Pipeline   ")
    print("==============================================")

    base_dir = Path(__file__).parent.resolve()
    raw_dir = base_dir / "data" / "raw"
    pre_dir = base_dir / "data" / "pre"
    processed_dir = base_dir / "data" / "processed"

    # Step 1: Pre-processing (Crop, Blur, CLAHE)
    print(f"\n[STEP 1/2] Pre-processing from {raw_dir.absolute()}...")
    preprocess_images(str(raw_dir), str(pre_dir))

    # Step 2: Augmentation (Flip, Rotate, etc.)
    print(f"\n[STEP 2/2] Augmentasi from {pre_dir}...")
    augment_images(str(pre_dir), str(processed_dir), num_augmentations_per_image=5)

    print("\n==============================================")
    print(f"Pipeline Selesai! Data siap digunakan di: {processed_dir}")
    print("==============================================")

if __name__ == "__main__":
    main()
