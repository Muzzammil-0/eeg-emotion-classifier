This project uses the EEG brainwave dataset by Jordan J. Bird et al.
Download it from: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions



## NEEDS MORE EEG DATA TO MAKE THE MODEL PERFORM BETTER AS IT CURRENTLY WORKS GOOD ONLY ON JORDAN J. BIRD'S DATASET


    # EEG Emotion Classifier

A Flask web application that classifies human emotions — **POSITIVE**, **NEUTRAL**, **NEGATIVE** — from EEG recordings in EDF format.

The core idea: most EEG emotion research is device-specific. This classifier normalises signals from **20+ different EEG headsets** to a common 4-electrode feature space, so one trained model works regardless of which device recorded the data. Upload any EDF file, get an emotion prediction.

Built as a solo mini-project in 4th semester CSE, SSTC Bhilai.



## How It Works

1. Upload an `.edf` file via the web interface
2. The pipeline **auto-detects** the EEG device from channel names
3. Channels are mapped to four canonical positions: `TP9 · AF7 · AF8 · TP10`
4. Signal is **filtered** (50 Hz notch + 0.5–45 Hz bandpass) and **resampled** to 150 Hz
5. FFT is computed → 500 frequency bins → converted to brainwave band features (delta, theta, alpha, beta, gamma + asymmetry/ratio features)
6. A **soft-voting ensemble** (Decision Tree + Random Forest + XGBoost) predicts the emotion
7. Result is shown in the browser

### Dual-Baseline Inference

Since we don't know the gender of a new subject, inference runs twice:
- Once treating the signal as the "male" side, paired against the female training baseline
- Once treating it as the "female" side, paired against the male baseline

Both probability vectors are averaged before the final prediction. This keeps asymmetry features meaningful for any input.



## Results

Trained on Jordan J. Bird's EEG Brainwave Dataset (2 subjects, Muse headset).

| Metric | Value |
|---|---|
| Train accuracy | 94.47% |
| Test accuracy | 88.79% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| NEGATIVE | 0.81 | 0.92 | 0.86 |
| NEUTRAL | 0.96 | 0.98 | 0.97 |
| POSITIVE | 0.88 | 0.74 | 0.80 |

> **Note:** The training dataset contains only 2 subjects. These numbers reflect within-dataset performance. Cross-subject generalisation has not yet been validated and is the next research goal.
# Needs more emotion labelled data to fulfill cross subject validation❤️


## Supported Devices

The channel registry currently handles:

| Brand | Models |
|---|---|
| Muse | 2016 4-channel |
| BioSemi | Active2 (64ch), 32ch |
| g.tec | g.USBamp 32, Nautilus, Sahara |
| EGI | GES 400 — 32 / 64 / 128 / 256 ch |
| Natus | Quantum LTM, EMU40EX |
| Neuroscan | 32ch, 64ch, NuAmps, SynAmps RT |
| Brain Products | actiCHamp 32, actiCHamp 64 |
| Nihon Kohden | 25ch, 64ch |
| PhysioNet | BCI2000 64ch motor |
| Generic fallback | Any standard 10-20 system device |

If your device isn't listed, it falls back to generic 10-20 channel matching automatically.

---

## Project Structure

```
eeg-emotion-classifier/
│
├── dipps.py                      # Core ML pipeline — signal processing, feature extraction, inference, training
├── sdf.py                        # Flask web app — entry point
├── model_utility_eeg.py          # Save / load / list versioned model snapshots
├── retraining_eeg_version.py     # CLI tool to retrain a saved version
├── compare_versions.py           # Compare accuracy across model versions
├── doctor_validation_set_...py   # Validation set evaluation script
│
├── IGT_dataset/                  # EDF files used for IGT participant training
├── templates/                    # HTML templates (index.html, result.html)
│
├── Dockerfile
├── Procfile                      # For Heroku / Railway deployment
└── requirements.txt
```

> Model artefacts (`.pkl`, `.npy`) and the training CSV are not included in this repo — see Dataset section below.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Muzzammil-0/eeg-emotion-classifier.git
cd eeg-emotion-classifier
pip install -r requirements.txt
```

### 2. Get the dataset

Download **EEG Brainwave Dataset: Feeling Emotions** by Jordan J. Bird:  
👉 https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

Place `emotions.csv` in the project root.

### 3. Train the model

```bash
python dipps.py --data emotions.csv --version version_0
```

This generates `model_version_0.pkl`, `label_encoder_version_0.pkl`, `male_baseline_version_0.npy`, `female_baseline_version_0.npy`, and `metrics_version_0.json`.

### 4. Run the web app

```bash
python sdf.py
```

Open `http://127.0.0.1:5000`, upload an EDF file, get a prediction.

### 5. Run with Docker

```bash
docker build -t eeg-classifier .
docker run -p 5000:5000 -e MODEL_VERSION=version_0 eeg-classifier
```

---

## Retraining

Retrain a saved version on its own data (useful after adding new EDF samples):

```bash
python retraining_eeg_version.py version_0
# Saves retrained model as version_0_trained
```

List all available saved versions:

```python
from model_utility_eeg import list_available_versions
list_available_versions()
```

---

## Model Versioning

Each training run saves a complete snapshot: model weights, label encoder, baselines, training features, and a metrics JSON. Versions are identified by name string (e.g. `version_0`, `igt_P05_trained`). The active version loaded by the Flask app is set via environment variable:

```bash
export MODEL_VERSION=igt_P05_trained  # default
```

---

## Signal Processing Pipeline

```
EDF file
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
  └─> Emotion label
```

---

## Dataset

> **Jordan J. Bird, Andreas Mangar, Dermot Fagan, Diego Faria**  
> *EEG-based Emotion Recognition Using a Deep Learning Network*  
> Dataset: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

Bird's work shows that 4 EEG channels are sufficient for emotion classification and that allowing natural artefacts like blinking produces more generalisable models. This project builds on that foundation and extends it toward device-agnostic deployment.

---

## Roadmap

- [ ] Cross-subject validation on DEAP / SEED datasets (32+ subjects)
- [ ] Fix: compute left/right hemisphere FFT separately to preserve asymmetry end-to-end in training
- [ ] Per-channel feature extraction instead of pre-FFT averaging
- [ ] Ethics committee approval + multi-subject clinical data collection
- [ ] Package as a standalone desktop app for clinical evaluation
- [ ] Add confidence score to predictions

---

## Tech Stack

Python · Flask · scikit-learn · XGBoost · MNE · SciPy · NumPy · Docker

---

## License

MIT
