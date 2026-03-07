#!/usr/bin/env python3
"""
Test script untuk memverifikasi ConvNeXt V2 build tanpa error
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing ConvNeXt V2 Model Build")
print("=" * 60)

try:
    print("\n1. Importing ConvNeXt V2...")
    from models.convnext_v2 import (
        build_convnext_v2_tiny,
        build_convnext_v2_small,
        build_convnext_v2_base,
        build_convnext_v2_large,
        build_convnext_v2_huge
    )
    print("   ✓ Import successful")
    
    print("\n2. Building ConvNeXt V2 Tiny model...")
    model_tiny = build_convnext_v2_tiny(num_classes=3)
    print(f"   ✓ Model shape: input={model_tiny.input_shape}, output={model_tiny.output_shape}")
    
    print("\n3. Building ConvNeXt V2 Small model...")
    model_small = build_convnext_v2_small(num_classes=3)
    print(f"   ✓ Model built successfully")
    
    print("\n4. Building ConvNeXt V2 Base model...")
    model_base = build_convnext_v2_base(num_classes=3)
    print(f"   ✓ Model built successfully")
    
    print("\n5. Checking model summary for Tiny variant...")
    print("\n" + "=" * 60)
    model_tiny.summary()
    print("=" * 60)
    
    print("\n✓ ALL TESTS PASSED!")
    print("\nConvNeXt V2 is ready for training with skin disease dataset!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
