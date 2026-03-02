from flask import Flask, request, jsonify, render_template, redirect, url_for
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

# ===== LOAD MODEL =====
version = os.environ.get('MODEL_VERSION', 'version_1_trained')
print(f" Loading model version: '{version}'")

try:
    model_path = f'model_{version}.pkl'
    model = joblib.load(model_path)
    le = joblib.load(f'label_encoder_{version}.pkl')
    male_baseline = np.load(f'male_baseline_{version}.npy')
    female_baseline = np.load(f'female_baseline_{version}.npy')
    print(" Model loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    raise

# ===== WEB INTERFACE ROUTES =====
@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and show result"""
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Get prediction
        emotion = predict_emotion_from_edf_single(
            temp_path,
            model=model,
            le=le,
            male_baseline=male_baseline,
            female_baseline=female_baseline
        )
        os.unlink(temp_path)
        
        # Show result page
        return render_template('result.html', emotion=emotion, filename=file.filename)
    
    except Exception as e:
        os.unlink(temp_path)
        error_msg = str(e)
        print(f" Error: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return render_template('index.html', error=f'Prediction failed: {error_msg}')

# ===== API ROUTES (keep these for programmatic access) =====
@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for programmatic access"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
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

# ===== START SERVER =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f" Starting server on port {port}")
    print(f" Web interface: http://127.0.0.1:{port}")
    print(f" API endpoint: http://127.0.0.1:{port}/predict")
    app.run(host='0.0.0.0', port=port, debug=False)
    