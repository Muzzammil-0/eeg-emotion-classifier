from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import tempfile

# Import your EEG functions
from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

# ===== LOAD MODEL USING ENVIRONMENT VARIABLE =====
version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f" Loading model version: {version}")

try:
    model = joblib.load(f'model_{version}.pkl')
    le = joblib.load(f'label_encoder_{version}.pkl')
    male_baseline = np.load(f'male_baseline_{version}.npy')
    female_baseline = np.load(f'female_baseline_{version}.npy')
    print(" Model loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        emotion = predict_emotion_from_edf_single(
            temp_path,
            model=model,
            le=le,
            male_baseline=male_baseline,
            female_baseline=female_baseline
        )
        os.unlink(temp_path)
        return jsonify({'emotion': emotion})
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive'})

# ===== CRITICAL: Use PORT from environment =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render default is 10000
    print(f" Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production