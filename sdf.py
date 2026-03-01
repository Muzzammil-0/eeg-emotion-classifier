from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import tempfile
import glob
import traceback
import sys

# Import your EEG functions
from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)

# ===== DEBUGGING =====
print(" Debug: Starting sdf.py")
print(f" Current working directory: {os.getcwd()}")
print(f" Files in directory: {os.listdir('.')}")

# ===== LOAD MODEL =====
version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f" Loading model version: '{version}'")
print(f" Version type: {type(version)}")
print(f" Version length: {len(version)}")

# Check what model files exist
model_files = glob.glob("model_*.pkl")
print(f" Available model files: {model_files}")

# Try to load with explicit error handling
try:
    model_path = f'model_{version}.pkl'
    print(f" Attempting to load: {model_path}")
    
    if not os.path.exists(model_path):
        print(f" File does not exist: {model_path}")
        print(" Checking for similar files:")
        for f in model_files:
            print(f"   - {f}")
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    le = joblib.load(f'label_encoder_{version}.pkl')
    male_baseline = np.load(f'male_baseline_{version}.npy')
    female_baseline = np.load(f'female_baseline_{version}.npy')
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f" Error loading model: {e}")
    print(f" Version value was: '{version}'")
    raise

# ===== ROUTES =====
@app.route('/predict', methods=['POST'])
def predict():
    try:
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
            # Print full traceback to Render logs
            print(" Error in prediction:")
            traceback.print_exc(file=sys.stdout)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(" Outer error:")
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'EEG Emotion Classifier API is running'})

# ===== START SERVER =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f" Attempting to start server on port: {port}")
    print(f" PORT environment variable: '{os.environ.get('PORT', 'NOT SET')}'")
    app.run(host='0.0.0.0', port=port, debug=False)