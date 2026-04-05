from doctor_validation_set_for_eeg_model import add_patient_to_training, show_log
from retraining_eeg_version import retrain_version
import os
import sys
import traceback
from flask import Flask, request, jsonify
import tempfile
import joblib
import glob
import re   
import numpy as np
app = Flask(__name__)


def get_latest_trained_version():
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

#ADD PATIENT DATA (EDF + label)
#------------------------------------------------------------
@app.route('/add_patient', methods=['POST'])

def add_patient():

    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Missing file or label'}), 400
    file = request.files['file']
    label = request.form['label']
    if label not in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        return jsonify({'error': 'Invalid label'}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        file.save(tmp.name)
    temp_path = tmp.name
    try:
    # Use the current active version (without '_trained') as base dataset version
        base_version = version
        if base_version.endswith('_trained'):
            base_version = base_version[:-8]   # e.g. "version_0_trained" -> "version_0"
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
#------------------------------------------------------------
#RETRAIN MODEL ON A GIVEN DATASET VERSION
#------------------------------------------------------------
@app.route('/retrain', methods=['POST'])

def retrain():

    data = request.get_json()
    version_name = data.get('version')          # e.g. "version_1"
    if not version_name:
        return jsonify({'error': 'Missing version'}), 400
    try:
        trained_version = retrain_version(version_name)
        return jsonify({
            'message': f'Retraining complete. New model: {trained_version}',
            'trained_version': trained_version
        })
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return jsonify({'error': str(e)}), 500
#------------------------------------------------------------
#RELOAD LATEST TRAINED MODEL (without restarting the server)
#------------------------------------------------------------
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