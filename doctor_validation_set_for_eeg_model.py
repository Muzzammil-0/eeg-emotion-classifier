
from dipps import extract_features_for_training
from model_utility_eeg import load_training_state, save_training_state
import numpy as np
import os
import re

def get_next_version_number():
    import re, os
    existing = [f for f in os.listdir() if f.startswith('model_version_') and f.endswith('.pkl')]
    numbers = []
    for f in existing:
        m = re.search(r'version_(\d+)', f)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers) + 1 if numbers else 1

def add_patient_to_training(edf_path, observed_emotion, base_version='version_0'):
    """
    Add a new patient's data and create new version
    
    Parameters:
    - edf_path: path to patient EDF file
    - observed_emotion: doctor's label ('POSITIVE', 'NEUTRAL', 'NEGATIVE')
    - base_version: version to build upon
    """
    
    print(f"\n Adding new patient from: {edf_path}")
    print(f"   Label: {observed_emotion}")
    
    # 1. Load base version
    X_base, y_base, model, le, male_baseline, female_baseline = load_training_state(base_version)
    
    # 2. Extract features from new patient
    new_features = extract_features_for_training(edf_path)
    
    if new_features is None:
        print(" Failed to extract features")
        return
    
    # 3. Combine with existing data
    X_new = np.vstack([X_base, new_features.reshape(1, -1)])
    y_new = np.concatenate([y_base, [observed_emotion]])
    
    # 4. Create new version name
 
    new_version = (f'version_{get_next_version_number()}')
    
    # 5. Save new version (note: model not retrained yet)
    save_training_state(
        version_0 = new_version,
        X= X_new,
        y= y_new,
        model= model,  # Original model, not retrained
        le= le,
        male_baseline= male_baseline,
        female_baseline= female_baseline
    )
    
    print(f"\n New version '{new_version}' created with {len(X_new)} samples")
    print("   Now retrain the model with:")
    print(f"   python retrain_model.py {new_version}")
    
    return new_version
