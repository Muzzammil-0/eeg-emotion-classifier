"""
igt_to_training.py
Convert Iowa Gambling Task dataset to your model's training format
"""

import pandas as pd
import numpy as np
import os
from scipy.fft import rfft, rfftfreq
import mne
from dipps import (
    reduce_eeg_noise, 
    bins_to_waves, 
    CHANNEL_REGISTRY, 
    EEGChannelMapper, 
    detect_device_model
)
from model_utility_eeg import load_training_state, save_training_state

def process_igt_subject(subject_id, data_folder, target_fs=150):
    """
    Process one IGT subject and extract features + labels
    
    Parameters:
    - subject_id: e.g., 'P01', 'P02', etc.
    - data_folder: path to extracted IGT dataset
    
    Returns:
    - features_array: numpy array of wave features for each trial
    - labels_array: corresponding emotion labels
    """
    
    print(f"\n Processing subject {subject_id}")
    
    # Load EEG data
    eeg_path = os.path.join(data_folder, f"{subject_id}", "EEG.csv")
    igt_path = os.path.join(data_folder, f"{subject_id}", "IGT.csv")
    
    # IGT.csv contains trial-by-trial decisions and outcomes
    igt_data = pd.read_csv(igt_path)
    
    # Load raw EEG (your pipeline expects EDF, but we have CSV)
    # You'll need to adapt based on actual format
    # Option 1: If it's raw EEG signals
    eeg_signals = pd.read_csv(eeg_path).values  # shape: (timepoints, channels)
    
    # Convert to MNE Raw object for your pipeline
    ch_names = [f'CH{i+1}' for i in range(eeg_signals.shape[1])]  # Adjust based on actual channel names
    sfreq = 250  # Check actual sampling rate in documentation
    info = mne.create_info(ch_names, sfreq, ch_types=['eeg']*len(ch_names))
    raw = mne.io.RawArray(eeg_signals.T, info)
    
    # Apply your noise filtration
    raw_data = raw.get_data()
    filtered = reduce_eeg_noise(raw_data, sfreq=sfreq)
    raw = mne.io.RawArray(filtered, info)
    
    # Resample to your target
    if sfreq != target_fs:
        raw.resample(target_fs)
    
    # Extract features for each trial
    features_list = []
    labels_list = []
    
    for idx, trial in igt_data.iterrows():
        # Extract EEG segment around this trial
        # You'll need to know timing information from the dataset
        # This is a placeholder – adjust based on actual data structure
        trial_start = trial['EEG_start_sample']  # Adjust column name
        trial_end = trial_start + target_fs * 2  # 2 seconds per trial
        
        if trial_end > raw.n_times:
            continue
            
        trial_data = raw.get_data()[:, trial_start:trial_end]
        
        # Create temporary raw object for this trial
        trial_raw = mne.io.RawArray(trial_data, raw.info)
        df_trial = trial_raw.to_data_frame()
        
        # Your existing channel mapping and feature extraction
        cleaned_names = [ch.replace('.','').replace('-','') for ch in df_trial.columns]
        df_trial.columns = cleaned_names
        
        # Detect device (may need fallback)
        device_model = detect_device_model(cleaned_names)
        mapping = CHANNEL_REGISTRY.get(device_model, CHANNEL_REGISTRY['generic_10_20'])['mapping']
        
        # Extract channels
        mapper = EEGChannelMapper()
        selected = {}
        for muse_ch, options in mapping.items():
            found = False
            for opt in options:
                opt_clean = opt.replace('.','').replace('-','')
                if opt_clean in cleaned_names:
                    selected[muse_ch] = df_trial[opt_clean].values
                    found = True
                    break
            if not found:
                for ch in cleaned_names:
                    if mapper.find_muse_channel(ch) == muse_ch:
                        selected[muse_ch] = df_trial[ch].values
                        found = True
                        break
        
        # Average signals
        combined = np.mean([selected['TP9'], selected['AF7'], 
                            selected['AF8'], selected['TP10']], axis=0)
        
        # FFT
        fft_vals = np.abs(rfft(combined))
        freqs = rfftfreq(len(combined), 1/target_fs)
        max_bin = np.searchsorted(freqs, 50.0)
        fft_500 = fft_vals[:max_bin]
        
        if len(fft_500) < 500:
            fft_500 = np.pad(fft_500, (0, 500 - len(fft_500)))
        else:
            fft_500 = fft_500[:500]
        
        # Build feature row
        row = {}
        for i, val in enumerate(fft_500):
            row[f'fft_{i}_a'] = val
            row[f'fft_{i}_b'] = val
        
        # Convert to wave features
        df_row = pd.DataFrame([row])
        wave_features = bins_to_waves(df_row).iloc[0].values
        
        features_list.append(wave_features)
        
        # Generate label based on outcome
        # This is where you define the emotion mapping
        if trial['outcome'] in ['win_large', 'win']:
            labels_list.append('POSITIVE')
        elif trial['outcome'] in ['loss_large', 'loss']:
            labels_list.append('NEGATIVE')
        else:
            labels_list.append('NEUTRAL')
    
    return np.array(features_list), np.array(labels_list)

def add_igt_to_training(data_folder, start_subject='P01', end_subject='P59'):
    """
    Add multiple IGT subjects to your training pipeline
    """
    
    # Start with base version
    current_version = 'version_0'
    
    for subj_num in range(1, 60):  # 59 subjects
        subject_id = f'P{subj_num:02d}'
        
        print(f"\n{'='*60}")
        print(f"Processing {subject_id}")
        print('='*60)
        
        try:
            # Process subject
            features, labels = process_igt_subject(subject_id, data_folder)
            
            if len(features) == 0:
                print(f"No valid trials for {subject_id}, skipping")
                continue
            
            print(f"Extracted {len(features)} trials from {subject_id}")
            
            # Load current training state
            X_current, y_current, model, le, male_bl, female_bl = load_training_state(current_version)
            
            # Append new data
            X_new = np.vstack([X_current, features])
            y_new = np.concatenate([y_current, labels])
            
            # Create new version
            new_version = f'igt_subject_{subj_num}'
            
            save_training_state(
                version_0=new_version,
                X=X_new,
                y=y_new,
                model=model,  # Original model, not retrained yet
                le=le,
                male_baseline=male_bl,
                female_baseline=female_bl
            )
            
            print(f"✅ Created {new_version} with {len(X_new)} total samples")
            current_version = new_version
            
        except Exception as e:
            print(f"❌ Error processing {subject_id}: {e}")
            continue
    
    return current_version

if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = r"C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\IGT_dataset"  # Adjusting the path
    
    print("="*60)
    print("IGT DATASET IMPORT PIPELINE")
    print("="*60)
    print("\nThis script will:")
    print("1. Process each IGT subject")
    print("2. Extract trials and generate emotion labels")
    print("3. Create new versions in your training pipeline")
    
    final_version = add_igt_to_training(DATA_FOLDER)
    
    print(f"\n✅ Import complete! Latest version: {final_version}")
    print("\nNext steps:")
    print(f"   python retrain_model.py {final_version}")
    print("   python compare_versions.py version_0 {final_version}_trained")

