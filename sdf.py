from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import tempfile

from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f"🚀 Loading model version: {version}")

model = joblib.load(f'model_{version}.pkl')
le = joblib.load(f'label_encoder_{version}.pkl')
male_baseline = np.load(f'male_baseline_{version}.npy')
female_baseline = np.load(f'female_baseline_{version}.npy')
print("✅ Model loaded successfully")

# ... (rest of your routes: /predict, /health)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)