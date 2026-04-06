import pandas as pd
import numpy as np

BIRD_CSV     = r'C:\Users\hp\Documents\EEG_BRAIN_FEELINGS_PROJECTS\emotions.csv'
EEG27_CSV    = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\emotions_27.csv'
OUTPUT_CSV   = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\emotions_combined.csv'

# --- Load Bird dataset ---
bird_df = pd.read_csv(BIRD_CSV, skiprows=1, header=None)
with open(BIRD_CSV, 'r') as f:
    header_line  = f.readline().strip('# \n')
    column_names = header_line.split(',')
bird_df.columns = column_names

# Keep only fft columns + label
fft_a_cols = [f'fft_{i}_a' for i in range(500)]
fft_b_cols = [f'fft_{i}_b' for i in range(500)]
fft_cols   = fft_a_cols + fft_b_cols

bird_df = bird_df[fft_cols + ['label']].copy()
bird_df['subject_id'] = 'bird_combined'
print(f"Bird dataset: {bird_df.shape}")
print(bird_df['label'].value_counts())

# --- Load EEGEmotions-27 ---
eeg27_df = pd.read_csv(EEG27_CSV)
print(f"\nEEG27 dataset: {eeg27_df.shape}")
print(eeg27_df['label'].value_counts())

# --- Combine ---
combined = pd.concat([bird_df, eeg27_df], ignore_index=True)

# Final column order
combined = combined[fft_cols + ['subject_id', 'label']]

combined.to_csv(OUTPUT_CSV, index=False)

print(f"\nCombined: {combined.shape}")
print(combined['label'].value_counts())
print(f"Subjects: {combined['subject_id'].nunique()}")
print(f"Saved: {OUTPUT_CSV}")
