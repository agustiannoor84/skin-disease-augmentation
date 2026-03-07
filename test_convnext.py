import sys
sys.path.append('.')

try:
    from models.convnext_v2 import build_convnext_v2_tiny
    print("Import successful")
    m = build_convnext_v2_tiny(num_classes=3)
    print("Model built successfully")
    print(f"Input shape: {m.input_shape}")
    print(f"Output shape: {m.output_shape}")
    print("Test passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()