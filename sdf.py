import os 
import sys 
import traceback 
import tempfile 
import glob 
import re 
import numpy as np 
import pandas as pd
import time as time
import joblib 
from flask import Flask, request, jsonify 
from flask_cors import CORS 
from dipps import ( predict_emotion_from_edf_single, detect_device_model,
                    _select_channels_from_edf, bins_to_waves,
                      _pad_or_truncate, reduce_eeg_noise, 
                      _CHANNEL_CACHE, _MAX_CACHE_SIZE ) 
from scipy.fft import rfft, rfftfreq 
from collections import OrderedDict 

import mne

#Doctor‑related imports

from doctor_validation_set_for_eeg_model import add_patient_to_training 
from retraining_eeg_version import retrain_version

app = Flask(__name__) 
CORS(app)

#Helper: get latest trained model version

def get_latest_trained_version():
    trained = glob.glob('model_*_trained.pkl')
    if trained:
        numbers = []
        for f in trained:
            m = re.search(r'version_(\d+)_trained', f)
            if m:
                numbers.append((int(m.group(1)), f))
        if numbers:
            latest = max(numbers, key=lambda x: x[0])
            return latest[1].replace('model_', '').replace('.pkl', '')

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

#Load model and related artefacts

version = get_latest_trained_version()
print(f"Loading model version: '{version}'")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, f'model-{version}.pkl')
if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found. Please ensure the model is trained and the file exists.")
    sys.exit(1)


try: 
    model = joblib.load(f'model_{version}.pkl') 
    le = joblib.load(f'label_encoder_{version}.pkl') 
    male_baseline = np.load(f'male_baseline_{version}.npy') 
    female_baseline = np.load(f'female_baseline_{version}.npy') 
    print("Model loaded successfully") 
except Exception as e:
    print(f"Error loading model: {e}") 
    raise

scaler_path = f'scaler_{version}.pkl' 
if os.path.exists(scaler_path):
     scaler = joblib.load(scaler_path) 
     print(f"Scaler loaded from '{scaler_path}'") 
else: scaler = None 
print("No scaler found, proceeding without it")

#Shared inference helper (used by /upload)

def _run_inference(temp_path):
    global _CHANNEL_CACHE 
    if temp_path in _CHANNEL_CACHE:
        selected = _CHANNEL_CACHE[temp_path] 
        _CHANNEL_CACHE.move_to_end(temp_path) 
    else: 
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False) 
        if raw.info['sfreq'] != 150:
            raw.resample(150)

        raw_data = raw.get_data() * 1e6
        filtered_data = reduce_eeg_noise(raw_data, sfreq=150)
        raw = mne.io.RawArray(filtered_data, raw.info)

        df_signals = raw.to_data_frame()
        cleaned_names = [ch.replace('.', '').replace('-', '') for ch in df_signals.columns]
        df_signals.columns = cleaned_names

        device_model = detect_device_model(cleaned_names)
        print(f"Detected device: {device_model}")

        selected = _select_channels_from_edf(df_signals, cleaned_names, device_model)
        _CHANNEL_CACHE[temp_path] = selected
        if len(_CHANNEL_CACHE) > _MAX_CACHE_SIZE:
            _CHANNEL_CACHE.popitem(last=False)

    combined_signal = np.mean([selected['TP9'], selected['AF7'],
                           selected['AF8'], selected['TP10']], axis=0)
    fft_vals = np.abs(rfft(combined_signal))
    freqs = rfftfreq(len(combined_signal), 1 / 150)
    max_bin = np.searchsorted(freqs, 50.0)
    fft_500 = _pad_or_truncate(fft_vals[:max_bin])

    if scaler is not None:
        all_cols = [f'fft_{i}_a' for i in range(500)] + [f'fft_{i}_b' for i in range(500)]
        raw_row = np.concatenate([fft_500, fft_500]).reshape(1, -1)
        scaled = scaler.transform(pd.DataFrame(raw_row, columns=all_cols))[0]
        fft_a = scaled[:500]
        fft_b = scaled[500:1000]
    else:
        fft_a = fft_b = fft_500

    row_male = {f'fft_{i}_a': fft_a[i] for i in range(500)}
    row_male.update({f'fft_{i}_b': female_baseline[i] for i in range(500)})

    row_female = {f'fft_{i}_b': fft_b[i] for i in range(500)}
    row_female.update({f'fft_{i}_a': male_baseline[i] for i in range(500)})

    features_male = bins_to_waves(pd.DataFrame([row_male]))
    features_female = bins_to_waves(pd.DataFrame([row_female]))

    proba_male = model.predict_proba(features_male)[0]
    proba_female = model.predict_proba(features_female)[0]
    avg_proba = (proba_male + proba_female) / 2

    pred_idx = int(np.argmax(avg_proba))
    emotion = le.inverse_transform([pred_idx])[0]
    classes = list(le.classes_)

    return emotion, avg_proba, classes

#routes

@app.route('/') 
def home(): 
    return render_template('index.html')

@app.route('/upload', methods=['POST']) 
def upload_file(): 
    start_total = time.time() 
    if 'file' not in request.files: 
        return jsonify({'error': 'No file uploaded'}), 400 
    file = request.files['file'] 
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
    temp_path = tmp.name

    try:
        emotion, avg_proba, classes = _run_inference(temp_path)
        os.unlink(temp_path)
        return jsonify({
            'emotion': emotion,
            'confidence': float(round(float(np.max(avg_proba)) * 100, 1)),
            'probabilities': {
                classes[i]: float(round(float(avg_proba[i]) * 100, 1))
                for i in range(len(classes))
            },
            'processing_time': round(time.time() - start_total, 2),
            'model_version': version
        })
    except Exception as e:

        if os.path.exists(temp_path):
            os.unlink(temp_path)
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive', 'model_version': version, 'scaler': scaler is not None})

@app.route('/add_patient', methods=['POST']) 
def add_patient():
    if 'file' not in request.files or 'label' not in request.form: return jsonify({'error': 'Missing file or label'}), 400

    file = request.files['file']
    label = request.form['label']

    if label not in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        return jsonify({'error': 'Invalid label'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        base_version = version
        if base_version.endswith('_trained'):
            base_version = base_version[:-8]

        new_version = add_patient_to_training(temp_path, label, base_version=base_version)
        os.unlink(temp_path)

        if new_version is None:
            return jsonify({'error': 'Failed to add patient'}), 500

        return jsonify({
            'message': f'Patient added. New dataset version: {new_version}',
            'new_version': new_version
        })
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500
    
@app.route('/retrain', methods=['POST']) 
def retrain():
    data = request.get_json()
    version_name = data.get('version') 
    if not version_name:
        return jsonify({'error': 'Missing version'}), 400
    try:
        trained_version = retrain_version(version_name)
        return jsonify({ 'message': f'Retraining complete. New model: {trained_version}', 'trained_version': trained_version })
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e)}), 500
    
@app.route('/reload_model', methods=['POST'])
def reload_model():
    global model, le, male_baseline, female_baseline, scaler, version 
    try: 
        new_version = get_latest_trained_version()
        if new_version == version:
            return jsonify({'message': 'Already using latest model', 'version': version})
        
        model = joblib.load(f'model_{new_version}.pkl')
        le = joblib.load(f'label_encoder_{new_version}.pkl')
        male_baseline = np.load(f'male_baseline_{new_version}.npy')
        female_baseline = np.load(f'female_baseline_{new_version}.npy')

        scaler_path = f'scaler_{new_version}.pkl'
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        version = new_version

        return jsonify({
            'message': f'Model reloaded to {new_version}',
            'version': new_version,
            'scaler_used': scaler is not None
        })
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e)}), 500
    
#main entry point

if __name__ == '__main__':
    import time 
    import pandas as pd 
    from flask import render_template
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

    