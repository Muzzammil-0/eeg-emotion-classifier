# EEG Emotion Classifier

A **device‑agnostic** EEG emotion classifier that works with raw EDF files from any headset.  
Predicts **POSITIVE / NEUTRAL / NEGATIVE** using an ensemble of tree‑based models.  
Includes a **Flutter desktop app** for clinicians and a **central server** that aggregates anonymised features from multiple clinics to continuously improve the global model.

> **To our knowledge, the first open‑source pipeline that accepts a raw EDF file from any EEG device and returns an emotion prediction end‑to‑end – no manual preprocessing, no device‑specific setup.**

## Why This Matters

Depression and anxiety are leading causes of disability worldwide, yet clinical assessment still relies heavily on subjective self‑report. Objective, continuous monitoring of emotional valence from EEG could enable early detection, objective mood tracking, and non‑intrusive monitoring in high‑risk populations.  
The **NEGATIVE** class – the most clinically important – is the hardest to classify. Missing a NEGATIVE state is far costlier than a false positive.

## Performance

- **Cross‑subject (LOSO):** 38% ± 9% balanced accuracy – slightly above random guess (33.3%), demonstrating generalisation.
- **Within‑dataset (training subjects):** 66.3% balanced accuracy (90 subjects from two continents).
- The model is **designed to improve** as more labelled data from diverse populations is collected.

## Core Innovations

1. **Device‑agnostic channel mapping** – automatically maps any EEG headset (Muse, Emotiv, BioSemi, etc.) to four canonical positions (TP9, AF7, AF8, TP10) using a comprehensive channel registry.
2. **Dual‑baseline inference** – handles unknown gender by running two predictions (signal as left/right hemisphere) and averaging probabilities.
3. **Continuous learning** – clinicians can add new labelled EDF files, retrain the local model, and **sync anonymised features** to a central server. The server aggregates data from multiple clinics, retrains a global ensemble, and clients can download the improved model.
4. **Flutter desktop app** – native Windows GUI for easy use in clinical settings.

## How It Works

```

EDF file → MNE load → notch/bandpass filter → resample 150 Hz → auto‑detect device
→ map to 4 channels → average → FFT (500 bins, 0‑50 Hz) → 18 brainwave‑band features
→ dual‑baseline inference → emotion prediction

```

## Feature Engineering (18 features)

- Per‑band mean power (delta, theta, alpha, beta, gamma) for left/right hemispheres – 10 features
- Hemispheric asymmetry (_a – _b) – 5 features
- Hemispheric symmetry ((_a + _b)/2) – 5 features
- Alpha ratio – 1 feature
- Theta/beta ratio per hemisphere – 2 features

## Ensemble Model

Soft‑voting ensemble of three classifiers with weights [1,2,5]:
- Decision Tree (balanced)
- Random Forest (balanced)
- XGBoost (regularised)

Hyperparameters selected via 5‑fold Stratified GridSearchCV optimising balanced accuracy.

## Supported Devices (20+ configurations)

| Brand | Models |
|-------|--------|
| Muse | 2016 4‑channel |
| Emotiv | EPOC X 14‑channel |
| BioSemi | Active2 (64ch), 32ch |
| g.tec | g.USBamp 32, Nautilus, Sahara |
| EGI | GES 400 – 32 / 64 / 128 / 256 ch |
| Natus | Quantum LTM, EMU40EX |
| Neuroscan | 32ch, 64ch, NuAmps, SynAmps RT |
| Brain Products | actiCHamp 32, actiCHamp 64 |
| Nihon Kohden | 25ch, 64ch |
| PhysioNet | BCI2000 64ch motor |
| Generic fallback | Any standard 10‑20 system |

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Muzzammil-0/eeg-emotion-classifier.git
cd eeg-emotion-classifier
pip install -r requirements.txt
```

2. Obtain datasets (optional for training)

· Bird dataset (required for initial model): Download from Kaggle – place emotions.csv in the project root.
· EEGEmotions-27 (optional, for combined training):
  ```bash
  git clone https://github.com/huytungst/EEGEmotions-27.git
  python preprocess_eeg27.py --input EEGEmotions-27/eeg_raw --output emotions_27.csv
  python combine_datasets.py
  ```

3. Train the initial model

```bash
# Bird dataset only
python dipps.py --data emotions.csv --version version_0

# Combined dataset (recommended)
python dipps.py --data emotions_combined.csv --version version_1
```

This generates:

· model_version_X.pkl
· label_encoder_version_X.pkl
· male_baseline_version_X.npy
· female_baseline_version_X.npy
· X_train_version_X.npy, y_train_version_X.npy, etc.

4. Run the central server (for multi‑clinic collaboration)

The central server aggregates anonymised features, retrains a global model, and serves it to clients.

```bash
python central_server.py
```

· Listens on port 5000 by default.
· Uses API key authentication (change API_KEY in the script).
· Endpoints: /upload_features, /retrain_global, /download_model, /download_label_encoder.

5. Run the local backend (sdf.py) and Flutter app

Local backend (handles EDF processing, local training, and sync):

```bash
python sdf.py
```

It runs on port 10000 (changeable). Make sure CENTRAL_URL in sdf.py points to your central server.

Flutter desktop app:

```bash
cd eeg_emotion_app
flutter pub get
flutter run -d windows
```

The app will connect to http://localhost:10000 (or your backend’s address).

6. Using the system

· Predict Emotion – select an EDF file, get an emotion label + confidence.
· Add Patient Data – label a new EDF file (POSITIVE/NEUTRAL/NEGATIVE). The local dataset expands.
· Retrain Model – retrains the local ensemble on the expanded dataset (creates a new version).
· Sync Local Data – uploads only the 18 anonymised features (not raw EDF) to the central server.
· Update Global Model – downloads the latest global model from the central server and replaces the local model.

7. Retrain the global model (on the central server)

After several clinics have synced their features, trigger a global retrain:

```bash
curl -X POST http://localhost:5000/retrain_global -H "X-API-Key: your-secret-key-123"
```

The central server will build a new global model using all aggregated features. Clients can then download it via the Update Global Model button.

Project Structure

```
eeg-emotion-classifier/
│
├── dipps.py                     # Core ML pipeline (training, inference, features)
├── sdf.py                       # Local backend (Flask, handles EDF uploads, sync)
├── central_server.py            # Central aggregation server
├── doctor_validation_set_for_eeg_model.py  # Add labelled EDFs to local dataset
├── retraining_eeg_version.py    # Retrain local model on current dataset
├── model_utility_eeg.py         # Save/load versioned model snapshots
├── preprocess_eeg27.py          # Convert EEGEmotions-27 TXT → CSV features
├── combine_datasets.py          # Merge Bird and EEGEmotions-27 CSVs
├── requirements.txt
├── Dockerfile
├── Procfile
├── eeg_emotion_app/             # Flutter desktop app
│   ├── lib/main.dart
│   ├── pubspec.yaml
│   └── windows/                 # Windows build output
└── templates/                   # HTML templates (for web interface)
```

Model Versioning & Continuous Learning

· Each training run saves a complete snapshot: model, label encoder, gender baselines, training features, and test set.
· Versions are named version_0, version_1, … and version_X_trained after retraining.
· The local backend always loads the latest trained version (get_latest_trained_version).
· The central server stores aggregated features in global_features.csv and produces global_model.pkl.

Limitations & Roadmap

Limitations

· Cross‑subject generalisation is still developing (38% LOSO). More diverse data will directly improve performance.
· 4‑channel abstraction loses spatial information; per‑channel features are planned.
· Asymmetry features are approximate when only one hemisphere signal is available.
· Label noise from 27→3 emotion mapping.

Roadmap

· ✅ Flutter desktop app
· ✅ Centralised feature aggregation + global model distribution
· ⬜ Federated learning (privacy‑preserving model averaging)
· ⬜ Real‑time inference from live EEG stream
· ⬜ Ethics approval + clinical data collection
· ⬜ Standalone Windows installer

Tech Stack

· Backend: Python, Flask, MNE, scikit‑learn, XGBoost, SciPy, NumPy
· Frontend: Flutter (Windows desktop)
· Deployment: Docker, Gunicorn
· Versioning: Custom file‑based snapshots

License

MIT


import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime
from functools import wraps

app = Flask(__name__)
CORS(app)

# ---------- Authentication ----------
API_KEY = "your-secret-key-123"   # MUST match the key used in sdf.py

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ---------- File paths ----------
DATA_CSV = 'global_features.csv'
MODEL_FILE = 'global_model.pkl'
LE_FILE = 'global_label_encoder.pkl'

# ---------- Helper ----------
def get_sample_count():
    if not os.path.exists(DATA_CSV):
        return 0
    try:
        df = pd.read_csv(DATA_CSV)
        return len(df)
    except:
        return 0

# ---------- Routes ----------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "alive", "samples": get_sample_count()})

@app.route('/upload_batch', methods=['POST'])
@require_api_key
def upload_batch():
    """Receive a batch of feature vectors + labels."""
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list of samples"}), 400
    if not data:
        return jsonify({"status": "ok", "uploaded": 0})

    n_features = len(data[0]['features'])
    rows = []
    for item in data:
        rows.append(item['features'] + [item['label']])

    df_new = pd.DataFrame(rows, columns=[f'f{i}' for i in range(n_features)] + ['label'])
    file_exists = os.path.exists(DATA_CSV) and os.path.getsize(DATA_CSV) > 0
    df_new.to_csv(DATA_CSV, mode='a', header=not file_exists, index=False)

    return jsonify({"status": "ok", "uploaded": len(data)})

@app.route('/retrain_global', methods=['POST'])
@require_api_key
def retrain_global():
    if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
        return jsonify({"error": "No data available"}), 400

    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, :-1].values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    rf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42)
    xgb = XGBClassifier(max_depth=3, n_estimators=50, random_state=42, n_jobs=-1)
    model = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 5]
    )
    model.fit(X, y_enc)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, LE_FILE)

    return jsonify({"message": f"Global model retrained on {len(df)} samples", "samples": len(df)})

@app.route('/download_model', methods=['GET'])
@require_api_key
def download_model():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "No global model yet"}), 404
    return send_file(MODEL_FILE, as_attachment=True)

@app.route('/download_label_encoder', methods=['GET'])
@require_api_key
def download_le():
    if not os.path.exists(LE_FILE):
        return jsonify({"error": "No label encoder yet"}), 404
    return send_file(LE_FILE, as_attachment=True)

@app.route('/stats', methods=['GET'])
@require_api_key
def stats():
    return jsonify({"samples": get_sample_count()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
