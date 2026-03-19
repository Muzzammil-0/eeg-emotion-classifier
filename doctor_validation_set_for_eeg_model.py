"""
Doctor Validation & Incremental Trainingw
=========================================
When a doctor provides an EDF file with a known emotion label,
this script adds it to the training data and creates a new version.

Workflow:
    1. Doctor provides EDF file + observed emotion label
    2. Run: python doctor_validation_set_for_eeg_model.py
    3. Then retrain: python retraining_eeg_version.py version_1

"""

from dipps import extract_features_for_training
from model_utility_eeg import load_training_state, save_training_state
import numpy as np
import os
import re
import shutil


def get_next_version_number():
    """Find the highest existing version number and return the next one."""
    existing = [f for f in os.listdir() if f.startswith('model_version_') and f.endswith('.pkl')]
    numbers  = []
    for f in existing:
        m = re.search(r'version_(\d+)', f)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers) + 1 if numbers else 1


def add_patient_to_training(edf_path, observed_emotion, base_version='version_0'):
    """
    Add a new patient EDF file to the training data and save a new version.

    Parameters
    ----------
    edf_path         : path to the patient's EDF file
    observed_emotion : doctor's observed label — 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'
    base_version     : the version to build upon (default: 'version_0')

    Notes
    -----
    This saves the expanded dataset as a new version but does NOT retrain the model yet.
    After running this, retrain with:
        python retraining_eeg_version.py version_<N>

    The scaler from base_version is copied to the new version so inference
    continues to work correctly after retraining.
    """
    print(f"\n  Adding new patient from: {edf_path}")
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
        print("  Failed to extract features from EDF — skipping.")
        return None

    # 3. Combine with existing data
    X_new = np.vstack([X_base, new_features.reshape(1, -1)])
    y_new = np.concatenate([y_base, [observed_emotion]])
    print(f"  Dataset expanded: {len(y_base)} → {len(y_new)} samples")

    # 4. Create new version name
    new_version = f'version_{get_next_version_number()}'

    # 5. Save new version (model not retrained yet — same model, new data)
    save_training_state(
        version_name=new_version,
        X=X_new,
        y=y_new,
        model=model,
        le=le,
        male_baseline=male_baseline,
        female_baseline=female_baseline
    )

    # 6. Copy scaler from base version so inference works after retraining
    scaler_src = f'scaler_{base_version}.pkl'
    scaler_dst = f'scaler_{new_version}.pkl'
    if os.path.exists(scaler_src):
        shutil.copy(scaler_src, scaler_dst)
        print(f"  Scaler copied: {scaler_dst}")
    else:
        print(f"  WARNING: scaler_{base_version}.pkl not found — inference may use unscaled FFT")

    print(f"\n  New version '{new_version}' created with {len(X_new)} samples")
    print(f"  Next step — retrain the model:")
    print(f"      python retraining_eeg_version.py {new_version}")

    return new_version


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Uncomment and update the call below when you have real EDF data
    # with a known emotion label from a doctor or clinical source.
    #
    # add_patient_to_training(
    #     edf_path='path/to/patient.edf',
    #     observed_emotion='NEGATIVE',   # or 'NEUTRAL' or 'POSITIVE'
    #     base_version='version_0'
    # )
    # ---------------------------------------------------------------

    print("No data to add yet.")
    print("When you have a labelled EDF file, uncomment the call above")
    print("and update the path and emotion label.")