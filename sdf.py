from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import tempfile
import glob
import re
import time
import traceback
import sys

from dipps import predict_emotion_from_edf_single, CHANNEL_REGISTRY, detect_device_model, bins_to_waves

app = Flask(__name__)


# ==============================================
# AUTO-DETECT LATEST MODEL VERSION
# ==============================================

def get_latest_trained_version():
    """Auto-detect the highest trained version available."""
    # Prefer *_trained versions first
    trained = glob.glob('model_*_trained.pkl')
    if trained:
        numbers = []
        for f in trained:
            m = re.search(r'version_(\d+)', f)
            if m:
                numbers.append((int(m.group(1)), f))
        if numbers:
            latest = max(numbers, key=lambda x: x[0])
            return latest[1].replace('model_', '').replace('.pkl', '')

    # Fall back to highest plain version
    plain = glob.glob('model_version_*.pkl')
    if plain:
        numbers = []
        for f in plain:
            m = re.search(r'version_(\d+)\.pkl$', f)
            if m:
                numbers.append((int(m.group(1)), f))
        if numbers:
            latest = max(numbers, key=lambda x: x[0])
            return latest[1].replace('model_', '').replace('.pkl', '')

    return 'version_0'


# ==============================================
# LOAD MODEL
# ==============================================

version = os.environ.get('MODEL_VERSION') or get_latest_trained_version()
print(f"  Loading model version: '{version}'")

try:
    model          = joblib.load(f'model_{version}.pkl')
    le             = joblib.load(f'label_encoder_{version}.pkl')
    male_baseline  = np.load(f'male_baseline_{version}.npy')
    female_baseline = np.load(f'female_baseline_{version}.npy')
    print("  Model loaded successfully")
except Exception as e:
    print(f"  Error loading model: {e}")
    raise

# Load scaler for the active version
scaler_path = f'scaler_{version}.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"  Scaler loaded: {scaler_path}")
else:
    scaler = None
    print(f"  WARNING: scaler not found for version '{version}' — inference will use raw FFT values")


# ==============================================
# ROUTES
# ==============================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    start_total = time.time()

    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected')

    t1 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    t2 = time.time()
    print(f"  File save: {t2 - t1:.2f}s")

    try:
        t3 = time.time()
        emotion = predict_emotion_from_edf_single(
            temp_path,
            model=model,
            le=le,
            male_baseline=male_baseline,
            female_baseline=female_baseline,
            scaler=scaler
        )
        t4 = time.time()
        print(f"  Prediction: {t4 - t3:.2f}s")

        os.unlink(temp_path)

        print(f"  Total: {time.time() - start_total:.2f}s")
        return render_template('result.html', emotion=emotion, filename=file.filename)

    except Exception as e:
        os.unlink(temp_path)
        error_msg = str(e)
        print(f"  Error: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return render_template('index.html', error=f'Prediction failed: {error_msg}')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive', 'model_version': version})


# ==============================================
# START SERVER
# ==============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"  Starting server on port {port}")
    print(f"  Web interface: http://127.0.0.1:{port}")
    print(f"  Upload endpoint: http://127.0.0.1:{port}/upload")
    app.run(host='0.0.0.0', port=port, debug=False)