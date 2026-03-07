import tensorflow as tf
import models.convnext_v2 as c
for name in ['tiny','small','base','large','huge']:
    fn = getattr(c, f'build_convnext_v2_{name}')
    print('Building', name)
    m = fn(num_classes=3)
    m.summary()
