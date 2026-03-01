from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import tempfile


print(" Files in current directory:", os.listdir('.'))

from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f" Loading model version: {version}")

# Check if file exists before loading
if os.path.exists(f'model_{version}.pkl'):
    print(f" Found model_{version}.pkl")
else:
    print(f" model_{version}.pkl NOT FOUND")
    print(" Available .pkl files:", [f for f in os.listdir('.') if f.endswith('.pkl')])

model = joblib.load(f'model_{version}.pkl')
le = joblib.load(f'label_encoder_{version}.pkl')
male_baseline = np.load(f'male_baseline_{version}.npy')
female_baseline = np.load(f'female_baseline_{version}.npy')
print("✅ Model loaded successfully")

# ... rest of your routes