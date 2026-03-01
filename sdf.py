from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import tempfile


print(" Files in current directory:", os.listdir('.'))

from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

# After your imports
import os
import glob

print(" Debug: Starting sdf.py")
print(f" Current working directory: {os.getcwd()}")
print(f" Files in directory: {os.listdir('.')}")

version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f" Loading model version: '{version}'")
print(f" Version type: {type(version)}")
print(f" Version length: {len(version)}")

# Check what model files exist
model_files = glob.glob('model_*.pkl')
print(f" Available model files: {model_files}")

# Try to load with explicit error handling
try:
    model_path = f'model_{version}.pkl'
    print(f" Attempting to load: {model_path}")
    
    if not os.path.exists(model_path):
        print(f" File does not exist: {model_path}")
        print(f" Checking for similar files:")
        for f in model_files:
            print(f"   - {f}")
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    le = joblib.load(f'label_encoder_{version}.pkl')
    male_baseline = np.load(f'male_baseline_{version}.npy')
    female_baseline = np.load(f'female_baseline_{version}.npy')
    print(" Model loaded successfully")
    
except Exception as e:
    print(f" Error loading model: {e}")
    print(f" Version value was: '{version}'")
    raise