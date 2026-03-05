import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))

from models.squeezenet import build_squeezenet

def test_build():
    print("Mencoba membangun SqueezeNet v1.1...")
    try:
        model = build_squeezenet(input_shape=(224, 224, 3), num_classes=2)
        print("Model berhasil dibangun!")
        model.summary()
    except Exception as e:
        print(f"Gagal membangun model: {e}")

if __name__ == "__main__":
    test_build()
