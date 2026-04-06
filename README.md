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


# ... (all imports and earlier code remain exactly the same) ...

# ---------- Central server configuration ----------
CENTRAL_URL = "http://localhost:5000"
CENTRAL_API_KEY = "your-secret-key-123"

# ---------- Sync tracking ----------
LAST_SYNC_FILE = 'last_sync_count.txt'

def get_last_sync_count():
    if os.path.exists(LAST_SYNC_FILE):
        with open(LAST_SYNC_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def set_last_sync_count(count):
    with open(LAST_SYNC_FILE, 'w') as f:
        f.write(str(count))

# ... (all other code up to the routes remains unchanged) ...

# ---------- Centralized aggregation endpoints ----------
# (Remove the old /get_local_features – we don't need it for batch upload)

@app.route('/sync_to_central', methods=['POST'])
def sync_to_central():
    global X_train, y_train
    if X_train is None or y_train is None:
        return jsonify({'error': 'No local training data'}), 400

    last_count = get_last_sync_count()
    total_count = len(X_train)
    if last_count >= total_count:
        return jsonify({'message': 'No new samples to sync'})

    # New samples since last sync
    new_features = X_train[last_count:].tolist()
    new_labels = y_train[last_count:].tolist()
    n_new = len(new_features)

    # Batch size (adjust as needed)
    batch_size = 1000
    uploaded = 0
    for i in range(0, n_new, batch_size):
        batch_feats = new_features[i:i+batch_size]
        batch_labels = new_labels[i:i+batch_size]
        payload = [{'features': f, 'label': l} for f, l in zip(batch_feats, batch_labels)]
        try:
            resp = requests.post(
                f'{CENTRAL_URL}/upload_batch',
                json=payload,
                headers={'X-API-Key': CENTRAL_API_KEY},
                timeout=30
            )
            if resp.status_code == 200:
                uploaded += len(batch_feats)
            else:
                return jsonify({'error': f'Batch upload failed: {resp.text}'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Update sync count
    set_last_sync_count(total_count)
    return jsonify({'message': f'Uploaded {uploaded} new samples (total {total_count})'})

# ... (the rest of the routes, including /download_global_model, remain exactly as before) ...
