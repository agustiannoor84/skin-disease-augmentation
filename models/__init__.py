# TensorFlow configuration to suppress dtype warnings
import os
import warnings

# Suppress warnings about missing dtypes in TensorFlow
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*float4_e2m1fn.*')

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
