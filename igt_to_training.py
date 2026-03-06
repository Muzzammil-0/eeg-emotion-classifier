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

def get_label_from_trial(trial, idx):
    """
    Generate emotion label based on real emotional significance
    """
    win = trial['win']
    lose = trial['lose']
    
    # Pure win (100) → POSITIVE
    if win >= 100 and lose == 0:
        return 'POSITIVE'
    
    # Pure loss (100) → NEGATIVE
    if lose >= 100 and win == 0:
        return 'NEGATIVE'
    
    # Mixed outcomes or small wins/losses → NEUTRAL
    if (win > 0 and lose > 0) or (win < 50 and lose < 50):
        return 'NEUTRAL'
    
    # Default fallback based on comparison
    if win > lose:
        return 'POSITIVE'
    elif lose > win:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

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
    
    print(f"\n🔍 Processing subject {subject_id}")
    
    # Check if subject folder exists
    subject_folder = os.path.join(data_folder, subject_id)
    if not os.path.exists(subject_folder):
        print(f"   ⚠️ Subject folder {subject_folder} not found, skipping")
        return None, None
    
    # Load EEG data
    eeg_path = os.path.join(subject_folder, "processed_EEG.csv")
    igt_path = os.path.join(subject_folder, "IGT.csv")
    
    # Check if files exist
    if not os.path.exists(eeg_path):
        print(f"   ⚠️ EEG file not found: {eeg_path}")
        return None, None
    if not os.path.exists(igt_path):
        print(f"   ⚠️ IGT file not found: {igt_path}")
        return None, None
    
    # Load data
    igt_data = pd.read_csv(igt_path)
    print(f"   📊 IGT columns: {igt_data.columns.tolist()}")
    print(f"   First few rows:\n{igt_data.head()}")
    
    # Load EEG signals
    print(f"   Loading EEG data from {eeg_path}")
    eeg_signals = pd.read_csv(eeg_path).values
    print(f"   EEG shape: {eeg_signals.shape}")
    
    # Create MNE Raw object
    ch_names = [f'CH{i+1}' for i in range(eeg_signals.shape[1])]
    sfreq = 250  # Original sampling rate from IGT dataset
    print(f"   Creating MNE Raw with {len(ch_names)} channels at {sfreq} Hz")
    
    info = mne.create_info(ch_names, sfreq, ch_types=['eeg']*len(ch_names))
    raw = mne.io.RawArray(eeg_signals.T, info)
    
    # Apply noise filtration
    print("   Applying noise filter...")
    raw_data = raw.get_data()
    filtered = reduce_eeg_noise(raw_data, sfreq=sfreq)
    raw = mne.io.RawArray(filtered, info)
    
    # Resample to your target
    if sfreq != target_fs:
        print(f"   Resampling from {sfreq} to {target_fs} Hz...")
        raw.resample(target_fs)
        sfreq = target_fs
    
    print(f"   Final data shape: {raw.get_data().shape}")
    
    # Extract features for each trial
    features_list = []
    labels_list = []
    
    # Calculate samples per trial (assuming 5-second trials at target_fs)
    samples_per_trial = int(5 * sfreq)
    print(f"   Using {samples_per_trial} samples per trial (5 seconds at {sfreq} Hz)")
    
    for idx, trial in igt_data.iterrows():
        try:
            # Get start sample from the 'EEG sample' column
            start_sample = int(trial['EEG sample'])
            
            # Calculate end sample
            end_sample = start_sample + samples_per_trial
            
            # Check if trial is within bounds
            if end_sample > raw.n_times:
                print(f"   ⚠️ Trial {idx} extends beyond recording (ends at {end_sample}, max {raw.n_times}), skipping")
                continue
            
            # Extract trial data
            trial_data = raw.get_data(start=start_sample, stop=end_sample)
            
            # Average across channels to get single signal
            combined_signal = np.mean(trial_data, axis=0)
            
            # Compute FFT (500 bins)
            fft_vals = np.abs(rfft(combined_signal))
            freqs = rfftfreq(len(combined_signal), 1/sfreq)
            max_bin = np.searchsorted(freqs, 50.0)
            fft_500 = fft_vals[:max_bin]
            
            # Pad/truncate to exactly 500
            if len(fft_500) < 500:
                fft_500 = np.pad(fft_500, (0, 500 - len(fft_500)))
            else:
                fft_500 = fft_500[:500]
            
            # Create row for wave features (A and B same signal)
            row = {}
            for i, val in enumerate(fft_500):
                row[f'fft_{i}_a'] = val
                row[f'fft_{i}_b'] = val
            
            df_row = pd.DataFrame([row])
            wave_features = bins_to_waves(df_row).iloc[0].values
            
            features_list.append(wave_features)
            
            # Generate label using the new function
            label = get_label_from_trial(trial, idx)
            labels_list.append(label)
                
        except Exception as e:
            print(f"   ❌ Error processing trial {idx}: {e}")
            continue
    
    if len(features_list) == 0:
        print(f"   No valid trials extracted for {subject_id}")
        return None, None
    
    print(f"   ✅ Extracted {len(features_list)} trials from {subject_id}")
    label_dist = pd.Series(labels_list).value_counts().to_dict()
    print(f"   Label distribution: {label_dist}")
    
    return np.array(features_list), np.array(labels_list)

def add_igt_to_training(data_folder, subject_list=None):
    """
    Add specific IGT subjects to your training pipeline
    
    Parameters:
    - data_folder: path to IGT dataset
    - subject_list: list of subjects to process, e.g. ['P01', 'P02', 'P05']
                    If None, processes all P01-P10
    """
    
    # Default to first 10 subjects if none specified
    if subject_list is None:
        subject_list = [f'P{i:02d}' for i in range(1, 11)]
    
    print(f"\n📋 Will process subjects: {subject_list}")
    
    # Start with base version
    current_version = 'version_0'
    
    for subject_id in subject_list:
        print(f"\n{'='*60}")
        print(f"📁 Processing {subject_id}")
        print('='*60)
        
        try:
            # Process subject
            features, labels = process_igt_subject(subject_id, data_folder)
            
            if features is None or len(features) == 0:
                print(f"   ⏭️ No valid trials for {subject_id}, skipping")
                continue
            
            # Load current training state
            X_current, y_current, model, le, male_bl, female_bl = load_training_state(current_version)
            
            # Append new data
            X_new = np.vstack([X_current, features])
            y_new = np.concatenate([y_current, labels])
            
            # Create new version
            new_version = f'igt_{subject_id}'
            
            save_training_state(
                version_0=new_version,
                X=X_new,
                y=y_new,
                model=model,
                le=le,
                male_baseline=male_bl,
                female_baseline=female_bl
            )
            
            print(f"   ✅ Created {new_version} with {len(X_new)} total samples")
            current_version = new_version
            
        except Exception as e:
            print(f"   ❌ Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return current_version

def check_available_subjects(data_folder):
    """Check which subjects are actually available in the folder"""
    available = []
    for i in range(1, 11):
        subj = f'P{i:02d}'
        if os.path.exists(os.path.join(data_folder, subj)):
            available.append(subj)
    return available

if __name__ == "__main__":
    # Configuration – update this to your actual path
    DATA_FOLDER = r"C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\IGT_dataset"
    
    # First check what's available
    available = check_available_subjects(DATA_FOLDER)
    
    print("="*60)
    print("🧠 IGT DATASET IMPORT PIPELINE")
    print("="*60)
    print(f"\n📁 Data folder: {DATA_FOLDER}")
    print(f"📊 Available subjects: {available}")
    
    if not available:
        print("\n❌ No subject folders found!")
        print("Please check:")
        print("1. The folder path is correct")
        print("2. You have downloaded and extracted the dataset")
        print("3. Subject folders are named P01, P02, etc.")
        exit()
    
    # Let user choose subjects
    print("\nWhich subjects do you want to process?")
    for i, subj in enumerate(available, 1):
        print(f"   {i}. {subj}")
    
    choice = input("\nEnter numbers (comma-separated) or 'all': ").strip()
    
    if choice.lower() == 'all':
        SUBJECTS_TO_PROCESS = available
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            SUBJECTS_TO_PROCESS = [available[i-1] for i in indices if 1 <= i <= len(available)]
        except:
            print("Invalid input, processing all available subjects")
            SUBJECTS_TO_PROCESS = available
    
    print(f"\n📋 Processing subjects: {SUBJECTS_TO_PROCESS}")
    
    final_version = add_igt_to_training(DATA_FOLDER, SUBJECTS_TO_PROCESS)
    
    print(f"\n✅ Import complete! Latest version: {final_version}")
    print("\n📌 Next steps:")
    print(f"   python retrain_model.py {final_version}")
    print(f"   python compare_versions.py version_0 {final_version}_trained")