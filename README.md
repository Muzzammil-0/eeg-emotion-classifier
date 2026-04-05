# EEG Emotion Classifier

A Flask web application that classifies human emotions — **POSITIVE**, **NEUTRAL**, **NEGATIVE** — directly from raw EEG recordings in EDF format.

> **To our knowledge, this is the first open-source pipeline that accepts a raw EDF file from any EEG headset and returns an emotion prediction end-to-end — no manual preprocessing, no device-specific setup.**

---

## UPDATE:** **From LOSO Cross-Validation: Achieved 38% +-9 % balanced accuracy across subjects -- highlighting the challenge of cross-subject generalization.
App Feature: One-click retraining from professional recordings. Creates versioned models (patient_john_v1.pkl) stored privately**

## Why This Matters

Depression and anxiety are among the leading causes of disability worldwide. Both conditions are characterised by sustained negative emotional states, yet clinical assessment still relies almost entirely on self-report — which is slow, subjective, and often inaccurate.

Objective, continuous monitoring of emotional valence from EEG could change this. A system that reliably flags sustained NEGATIVE emotional states opens a path toward:

- Early detection of depressive episodes before clinical threshold
- Objective mood tracking during psychiatric treatment
- Passive, non-intrusive monitoring in high-risk populations

The NEGATIVE class is the hardest to classify in EEG emotion research — and deliberately the most important class to get right. Missing a NEGATIVE state in a clinical context is a far costlier error than a false positive.

---

## Performance

The model performs **2× better than random chance** on a 3-class problem. A random classifier achieves 33.3% balanced accuracy by definition. This model achieves **66.3% balanced accuracy** across 90 subjects from two continents — without any subject-specific calibration.

### Training History

| Version | Dataset | Subjects | Train Acc | Test Acc | Notes |
|---------|---------|----------|-----------|----------|-------|
| v0 | Bird only | 2 | 94.47% | 88.79% | Within-dataset only, not generalisable |
| v1 | Combined, no weights | 90 | 73.87% | 70.54% | First honest cross-subject result |
| v2 | Combined + class weights | 90 | 76.35% | 66.29% | Improved NEGATIVE recall |
| v3 | Combined + resampling + grid search | 90 | 98.66% | 66.96% | Overfit |
| v4 | Combined + regularisation grid search | 90 | 80.10% | 66.07% | Healthiest train/test gap |

### Version 4 — Current Best (Cross-Subject)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NEGATIVE | 0.53 | 0.61 | 0.56 |
| NEUTRAL | 0.74 | 0.74 | 0.74 |
| POSITIVE | 0.72 | 0.65 | 0.69 |
| **Macro avg** | **0.67** | **0.67** | **0.67** |

Train/test gap of only 14 percentage points — the most generalised version to date.

### Version 0 — Single-Dataset Reference

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NEGATIVE | 0.81 | 0.92 | 0.86 |
| NEUTRAL | 0.96 | 0.98 | 0.97 |
| POSITIVE | 0.88 | 0.74 | 0.80 |

**Note:** v0 numbers reflect within-dataset performance on 2 subjects and should not be taken as a generalisation benchmark.

---

## The Core Idea

Most EEG emotion research is device-specific — a model trained on Muse data fails on BioSemi data, and vice versa, because channel names, layouts and sampling rates differ across manufacturers.

This classifier solves that by normalising any EEG recording to a **common 4-electrode feature space** (TP9 · AF7 · AF8 · TP10) regardless of the source device. One trained model works on input from any supported headset.

---

## How It Works

```
Upload .edf file
  └─> MNE load
  └─> Notch filter @ 50 Hz  +  Bandpass 0.5–45 Hz
  └─> Resample to 150 Hz
  └─> Auto-detect device from channel names
  └─> Map channels → [TP9, AF7, AF8, TP10]
  └─> Average 4 channels → 1D signal
  └─> FFT → 500 bins (0–50 Hz)
  └─> Brainwave bands: delta · theta · alpha · beta · gamma
      + asymmetry · symmetry · alpha ratio · theta/beta ratio
  └─> Dual-baseline inference (male branch + female branch → average)
  └─> Emotion label → browser
```

### Signal Processing

- **Notch filter at 50 Hz** removes powerline interference (IIR, Q=30)
- **Bandpass 0.5–45 Hz** (4th-order Butterworth) isolates physiologically relevant EEG bands
- **Resampling to 150 Hz** normalises recordings from different devices (e.g. Emotiv X at 256 Hz → 150 Hz) so FFT bin indices are comparable across datasets
- **FFT over 0–50 Hz** yields 500 frequency bins capturing delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) and gamma (30–45 Hz) bands

### Feature Engineering

From 500 raw FFT bins, 18 higher-level features are computed:

- Per-band mean power for left (_a) and right (_b) hemispheres: 10 features
- Per-band hemispheric asymmetry (_a − _b): 5 features
- Per-band hemispheric symmetry ((_a + _b) / 2): 5 features  
- Alpha lateralisation ratio: 1 feature
- Theta/beta ratio for each hemisphere: 2 features (marker for attention and emotional arousal)

### Dual-Baseline Inference

EEG asymmetry features are only meaningful when compared against a reference signal. Since the gender of a new subject is unknown at inference time, the pipeline runs two predictions:

1. Signal treated as left-hemisphere (_a), paired against the female training baseline as _b
2. Signal treated as right-hemisphere (_b), paired against the male training baseline as _a

Both probability vectors are averaged before argmax. This preserves the asymmetry signal for any input without requiring demographic metadata.

### Ensemble Model

Soft-voting ensemble of three classifiers with weights [1, 2, 5]:

- Decision Tree (shallow, class-balanced)
- Random Forest (class-balanced, min_samples_leaf=10)
- XGBoost (min_child_weight=10 for regularisation)

Class weights are balanced using `compute_sample_weight` to compensate for the POSITIVE-heavy label distribution in combined training data. Hyperparameters selected via 5-fold stratified GridSearchCV optimising balanced accuracy.

---

## Datasets

### Jordan J. Bird et al. — EEG Brainwave Dataset: Feeling Emotions
2 subjects, Muse 4-channel headset, 2132 samples, POSITIVE / NEUTRAL / NEGATIVE labels.

> Bird, J.J., Mangar, A., Fagan, D., Faria, D. (2019). EEG-based Emotion Recognition Using a Deep Learning Network.  
> Dataset: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

Bird's work demonstrates that 4 EEG channels are sufficient for emotion classification and that preserving natural artefacts produces more generalisable models. This project builds on that foundation.

### EEGEmotions-27
88 subjects, Emotiv X 14-channel headset, 256 Hz, 2342 samples after preprocessing. Originally annotated with 27 fine-grained emotion labels, collapsed to 3 classes for this project. Subjects are Korean and Vietnamese, ages 20s–60s.

> Phuong, H.T., Im, E.T., Oh, M.S., Gim, G.Y. (2025). EEGEmotions-27: A Large-Scale EEG Dataset Annotated With 27 Fine-Grained Emotion Labels. IEEE Access, 13, 176915–176932.  
> DOI: https://doi.org/10.1109/ACCESS.2025.3620677

**Combined training set: 4474 samples, 90 subjects, 2 continents, 2 headset types.**

EEGEmotions-27 signals are resampled from 256 Hz to 150 Hz before FFT so frequency bin indices are directly comparable to the Bird dataset.

---

## Supported Devices

The channel registry handles automatic mapping for 20+ device configurations:

| Brand | Models |
|-------|--------|
| Muse | 2016 4-channel |
| Emotiv | EPOC X 14-channel |
| BioSemi | Active2 (64ch), 32ch |
| g.tec | g.USBamp 32, Nautilus, Sahara |
| EGI | GES 400 — 32 / 64 / 128 / 256 ch |
| Natus | Quantum LTM, EMU40EX |
| Neuroscan | 32ch, 64ch, NuAmps, SynAmps RT |
| Brain Products | actiCHamp 32, actiCHamp 64 |
| Nihon Kohden | 25ch, 64ch |
| PhysioNet | BCI2000 64ch motor |
| Generic fallback | Any standard 10-20 system device |

Detection is automatic from channel names. If a device is not recognised, the pipeline falls back to generic 10-20 matching and attempts to find the nearest anatomical equivalents.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Muzzammil-0/eeg-emotion-classifier.git
cd eeg-emotion-classifier
pip install -r requirements.txt
```

### 2. Get the datasets

**Bird dataset (required):**  
Download from https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions  
Place `emotions.csv` in the project root.

**EEGEmotions-27 (optional, for combined training):**
```bash
git clone https://github.com/huytungst/EEGEmotions-27.git
python preprocess_eeg27.py --input EEGEmotions-27/eeg_raw --output emotions_27.csv
python combine_datasets.py
```

### 3. Train

```bash
# Bird dataset only
python dipps.py --data emotions.csv --version version_0

# Combined dataset (recommended)
python dipps.py --data emotions_combined.csv --version version_1
```

Generates: `model_version_X.pkl`, `label_encoder_version_X.pkl`, `male_baseline_version_X.npy`, `female_baseline_version_X.npy`, `metrics_version_X.json`

### 4. Run the web app

```bash
python sdf.py
```

Open http://127.0.0.1:5000, upload an EDF file, get a prediction.

### 5. Docker

```bash
docker build -t eeg-classifier .
docker run -p 5000:5000 -e MODEL_VERSION=version_0 eeg-classifier
```

---

## Project Structure

```
eeg-emotion-classifier/
│
├── dipps.py                    # Core ML pipeline — signal processing, feature extraction, inference, training
├── sdf.py                      # Flask web app — entry point
├── model_utility_eeg.py        # Save / load / list versioned model snapshots
├── preprocess_eeg27.py         # Convert EEGEmotions-27 txt files to FFT feature CSV
├── combine_datasets.py         # Concatenate Bird + EEGEmotions-27 CSVs
├── retraining_eeg_version.py   # CLI tool to retrain a saved version
├── compare_versions.py         # Compare accuracy across model versions
├── doctor_validation_set_...py # Validation set evaluation script
│
├── IGT_dataset/                # EDF files used for IGT participant training
├── templates/                  # HTML templates (index.html, result.html)
│
├── Dockerfile
├── Procfile                    # For Heroku / Railway deployment
└── requirements.txt
```

Model artefacts (`.pkl`, `.npy`) and training CSVs are not included — see Dataset section above.

---

## Model Versioning

Each training run saves a complete snapshot: model weights, label encoder, gender baselines, training features, and a metrics JSON. Versions are identified by name string.

```python
from model_utility_eeg import list_available_versions
list_available_versions()
```

Set the active version via environment variable:

```bash
export MODEL_VERSION=version_1
```

Retrain a saved version on its own data:

```bash
python retraining_eeg_version.py version_0
```

---

## Limitations

- **Cross-subject generalisation is still developing.** 90 subjects is meaningful progress but not clinical-grade. More diverse data will directly improve performance.
- **4-channel abstraction loses spatial information.** Mapping 64-channel recordings down to 4 canonical positions discards inter-electrode relationships. Per-channel feature extraction is on the roadmap.
- **Asymmetry features are approximate in training.** The Bird dataset uses one male and one female subject paired in _a/_b columns. EEGEmotions-27 signals are duplicated into both columns since individual-level pairing is unavailable. This is partially compensated by dual-baseline inference.
- **Label noise from 27→3 mapping.** Collapsing 27 fine-grained emotions into 3 categories introduces ambiguity — "craving" as NEUTRAL and "nostalgia" as POSITIVE are debatable assignments.

---

## Roadmap

- [ ] Confidence scores + uncertainty
- [ ] DEAP dataset integration (32 subjects, valence/arousal → 3-class mapping)
- [ ] Per-channel FFT instead of pre-averaging to preserve spatial information
- [ ] Compute left/right hemisphere FFT separately end-to-end in training
- [ ] Confidence score on predictions
- [ ] Real-time inference from live EEG stream
- [ ] Ethics committee approval + clinical data collection
- [ ] Standalone desktop app for clinical evaluation

---

## Tech Stack

Python · Flask · scikit-learn · XGBoost · MNE · SciPy · NumPy · Docker

---

## License

MIT

import os
import sys
import traceback
import tempfile
import glob
import re
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dipps import (
    predict_emotion_from_edf_single,
    detect_device_model,
    _select_channels_from_edf,
    bins_to_waves,
    _pad_or_truncate,
    reduce_eeg_noise,
    _CHANNEL_CACHE,
    _MAX_CACHE_SIZE
)
from scipy.fft import rfft, rfftfreq
from collections import OrderedDict
import mne

# Doctor‑related imports
from doctor_validation_set_for_eeg_model import add_patient_to_training
from retraining_eeg_version import retrain_version

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# Helper: get latest trained model version
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Load model and related artefacts
# ----------------------------------------------------------------------
version = get_latest_trained_version()
print(f"Loading model version: '{version}'")

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
else:
    scaler = None
    print("No scaler found, proceeding without it")

# ----------------------------------------------------------------------
# Shared inference helper (used by /upload)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
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
        return jsonify({
            'message': f'Retraining complete. New model: {trained_version}',
            'trained_version': trained_version
        })
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

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import time
    import pandas as pd
    from flask import render_template

    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
