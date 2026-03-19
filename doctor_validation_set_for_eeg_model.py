"""
Doctor Validation & Incremental Training
=========================================
When a doctor provides EDF files with known emotion labels,
this script adds them to the training data and creates new versions.

Workflow:
    1. Doctor provides EDF file(s) + observed emotion label(s)
    2. Run: python doctor_validation_set_for_eeg_model.py
    3. Then retrain: python retraining_eeg_version.py

Single patient:
    add_patient_to_training('patient.edf', 'NEGATIVE')

Multiple patients at once:
    add_multiple_patients([
        ('patient1.edf', 'NEGATIVE'),
        ('patient2.edf', 'POSITIVE'),
        ('patient3.edf', 'NEUTRAL'),
    ])

Check what has been added:
    show_log()
"""

from dipps import extract_features_for_training
from model_utility_eeg import load_training_state, save_training_state
import numpy as np
import os
import re
import shutil
import json
from datetime import datetime
from collections import Counter


# ==============================================
# HELPERS
# ==============================================

def get_next_version_number():
    """Find the highest existing version number and return the next one."""
    existing = [f for f in os.listdir() if f.startswith('model_version_') and f.endswith('.pkl')]
    numbers  = []
    for f in existing:
        m = re.search(r'version_(\d+)', f)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers) + 1 if numbers else 1


def log_patient_entry(edf_path, emotion, version, log_file='patient_log.json'):
    """
    Append an entry to the patient log so you can track what was added,
    when, with what label, and which version it went into.
    """
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'edf_path':  edf_path,
        'emotion':   emotion,
        'version':   version,
    }

    log = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = json.load(f)

    log.append(entry)

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"  Logged to {log_file}")


def show_log(log_file='patient_log.json'):
    """Print a summary of all patients added so far."""
    if not os.path.exists(log_file):
        print("  No log file found — no patients added yet.")
        return

    with open(log_file, 'r') as f:
        log = json.load(f)

    print(f"\n  Total patients logged: {len(log)}")

    label_counts = Counter(e['emotion'] for e in log)
    print(f"  Label breakdown:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

    print(f"\n  Last 5 entries:")
    for entry in log[-5:]:
        print(f"    [{entry['timestamp']}] {entry['emotion']} — {os.path.basename(entry['edf_path'])} → {entry['version']}")


# ==============================================
# CORE FUNCTIONS
# ==============================================

def add_patient_to_training(edf_path, observed_emotion, base_version='version_0'):
    """
    Add a single patient EDF file to the training data and save a new version.

    Parameters
    ----------
    edf_path         : path to the patient's EDF file
    observed_emotion : doctor's observed label — 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'
    base_version     : the version to build upon (default: 'version_0')

    Notes
    -----
    This saves the expanded dataset as a new version but does NOT retrain yet.
    After running this (or add_multiple_patients), retrain with:
        python retraining_eeg_version.py
    """
    print(f"\n  Adding patient from: {edf_path}")
    print(f"  Label: {observed_emotion}")

    valid_emotions = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    if observed_emotion not in valid_emotions:
        print(f"  ERROR: observed_emotion must be one of {valid_emotions}")
        return None

    # 1. Load base version
    X_base, y_base, model, le, male_baseline, female_baseline = load_training_state(base_version)
    print(f"  Base version loaded: {len(y_base)} existing samples")

    # 2. Extract features from new patient EDF
    new_features = extract_features_for_training(edf_path)
    if new_features is None:
        print("  Failed to extract features — skipping.")
        return None

    # 3. Combine with existing data
    X_new = np.vstack([X_base, new_features.reshape(1, -1)])
    y_new = np.concatenate([y_base, [observed_emotion]])
    print(f"  Dataset expanded: {len(y_base)} → {len(y_new)} samples")

    # 4. Create new version name
    new_version = f'version_{get_next_version_number()}'

    # 5. Save new version (model not retrained yet)
    save_training_state(
        version_name=new_version,
        X=X_new,
        y=y_new,
        model=model,
        le=le,
        male_baseline=male_baseline,
        female_baseline=female_baseline
    )

    # 6. Copy scaler so inference works after retraining
    scaler_src = f'scaler_{base_version}.pkl'
    scaler_dst = f'scaler_{new_version}.pkl'
    if os.path.exists(scaler_src):
        shutil.copy(scaler_src, scaler_dst)
        print(f"  Scaler copied: {scaler_dst}")
    else:
        print(f"  WARNING: scaler_{base_version}.pkl not found")

    # 7. Log the entry
    log_patient_entry(edf_path, observed_emotion, new_version)

    print(f"\n  Version '{new_version}' created with {len(X_new)} samples")
    print(f"  Retrain when ready: python retraining_eeg_version.py {new_version}")

    return new_version


def add_multiple_patients(patient_list, base_version='version_0'):
    """
    Add multiple labelled EDF files at once before retraining.
    Each patient chains into the next version automatically.

    Parameters
    ----------
    patient_list : list of tuples — [(edf_path, emotion_label), ...]
    base_version : version to build upon

    Example
    -------
    add_multiple_patients([
        ('patient1.edf', 'NEGATIVE'),
        ('patient2.edf', 'POSITIVE'),
        ('patient3.edf', 'NEUTRAL'),
    ])
    """
    print(f"\n  Adding {len(patient_list)} patients starting from {base_version}...")

    current_version = base_version
    added           = 0
    failed          = []

    for edf_path, emotion in patient_list:
        result = add_patient_to_training(edf_path, emotion, base_version=current_version)
        if result is not None:
            current_version = result
            added += 1
        else:
            failed.append((edf_path, emotion))

    print(f"\n  ===== Batch complete =====")
    print(f"  Added:  {added}/{len(patient_list)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for path, label in failed:
            print(f"    - {os.path.basename(path)} [{label}]")
    print(f"  Final version: {current_version}")
    print(f"  Now retrain:   python retraining_eeg_version.py {current_version}")

    return current_version


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # SINGLE PATIENT — uncomment when you have real labelled EDF data
    #
    # add_patient_to_training(
    #     edf_path='path/to/patient.edf',
    #     observed_emotion='NEGATIVE',
    #     base_version='version_0'
    # )
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # MULTIPLE PATIENTS — uncomment when you have a batch of EDF files
    #
    # add_multiple_patients([
    #     ('path/to/patient1.edf', 'NEGATIVE'),
    #     ('path/to/patient2.edf', 'POSITIVE'),
    #     ('path/to/patient3.edf', 'NEUTRAL'),
    # ], base_version='version_0')
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # CHECK LOG — see what has been added so far
    #
    # show_log()
    # ---------------------------------------------------------------

    print("No data to add yet.")
    print("Uncomment one of the calls above when you have labelled EDF files.")
    print("Run show_log() to see what has been added so far.")