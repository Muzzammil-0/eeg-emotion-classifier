from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys
import tempfile
import os

# Import your EEG functions 
from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

# ===== LOAD MODEL BASED ON COMMAND LINE ARGUMENT =====
version = sys.argv[1] if len(sys.argv) > 1 else 'version_1_trained'
print(f" Loading model version: {version}")

model = joblib.load(f'model_{version}.pkl')
le = joblib.load(f'label_encoder_{version}.pkl')
male_baseline = np.load(f'male_baseline_{version}.npy')
female_baseline = np.load(f'female_baseline_{version}.npy')
print(" Model loaded successfully")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Save uploaded file temporarily
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
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive'})

# ===== THIS LINE ACTUALLY STARTS THE SERVER =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)