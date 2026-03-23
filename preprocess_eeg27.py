import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy import signal
from pathlib import Path
import os

# ==============================================
# EMOTION MAPPING 27 -> 3
# ==============================================

EMOTION_MAP = {
    1:  'POSITIVE',  # admiration
    2:  'POSITIVE',  # adoration
    3:  'POSITIVE',  # aesthetic
    4:  'POSITIVE',  # amusement
    7:  'POSITIVE',  # awe
    10: 'POSITIVE',  # calmness
    15: 'POSITIVE',  # entrancement
    16: 'POSITIVE',  # excitement
    19: 'POSITIVE',  # interest
    20: 'POSITIVE',  # joy
    21: 'POSITIVE',  # nostalgia
    22: 'POSITIVE',  # relief
    23: 'POSITIVE',  # romance
    25: 'POSITIVE',  # satisfaction
    27: 'POSITIVE',  # surprised
    5:  'NEGATIVE',  # anger
    6:  'NEGATIVE',  # anxiety
    8:  'NEGATIVE',  # awkwardness
    13: 'NEGATIVE',  # disgust
    14: 'NEGATIVE',  # empathic pain
    17: 'NEGATIVE',  # fear
    18: 'NEGATIVE',  # horror
    24: 'NEGATIVE',  # sadness
    9:  'NEUTRAL',   # boredom
    11: 'NEUTRAL',   # confusion
    12: 'NEUTRAL',   # craving
    26: 'NEUTRAL',   # sexual desire
}

SFREQ = 256  # Emotiv X sampling rate

# ==============================================
# SIGNAL PROCESSING
# ==============================================

def reduce_eeg_noise(eeg_signal, sfreq=256, notch_freq=50, lowcut=0.5, highcut=45):
    filtered = eeg_signal.copy()
    nyq = 0.5 * sfreq
    notch = notch_freq / nyq
    b_notch, a_notch = signal.iirnotch(notch, Q=30)
    low  = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = signal.butter(4, [low, high], btype='band')
    for ch in range(filtered.shape[0]):
        filtered[ch] = signal.filtfilt(b_notch, a_notch, filtered[ch])
        filtered[ch] = signal.filtfilt(b_band,  a_band,  filtered[ch])
    return filtered

def _pad_or_truncate(arr, target=500):
    if len(arr) >= target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)))

def process_file(filepath, sfreq=SFREQ, target_bins=500):
    """Read one EEGEmotions-27 txt file and return 500-bin FFT array."""
    data = np.loadtxt(filepath)        # shape: (n_samples, 14)
    data = data.T                      # shape: (14, n_samples)
    data = reduce_eeg_noise(data, sfreq=sfreq)
    combined = data.mean(axis=0)       # average across 14 channels

    fft_vals = np.abs(rfft(combined))
    freqs    = rfftfreq(len(combined), 1 / sfreq)
    max_bin  = np.searchsorted(freqs, 50.0)
    fft_500  = _pad_or_truncate(fft_vals[:max_bin], target=target_bins)

    return fft_500

# ==============================================
# MAIN CONVERSION
# ==============================================

def build_emotions27_csv(eeg_raw_dir, output_csv='emotions_27.csv'):
    eeg_path = Path(eeg_raw_dir)
    files    = sorted(eeg_path.glob('*.txt'))
    print(f"Found {len(files)} files in {eeg_raw_dir}")

    rows   = []
    errors = []

    for i, filepath in enumerate(files):
        # Parse filename: {participant_id}_{emotion_id}.0.txt
        try:
            stem       = filepath.stem          # e.g. "10_1.0"
            parts      = stem.split('_')
            p_id       = int(parts[0])
            e_id       = int(parts[1].split('.')[0])
            label      = EMOTION_MAP.get(e_id)

            if label is None:
                print(f"  Skipping unknown emotion id {e_id} in {filepath.name}")
                continue

            fft_500 = process_file(filepath)

            # Duplicate into _a and _b to match Bird dataset format
            row = {f'fft_{j}_a': fft_500[j] for j in range(500)}
            row.update({f'fft_{j}_b': fft_500[j] for j in range(500)})
            row['label']       = label
            row['participant'] = p_id
            row['emotion_id']  = e_id
            rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(files)}")

        except Exception as e:
            errors.append(filepath.name)
            print(f"  ERROR in {filepath.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*50}")
    print(f"Done! {len(df)} rows saved to {output_csv}")
    print(f"Errors: {len(errors)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nParticipants: {df['participant'].nunique()}")
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=r"C:\Users\hp\Documents\EEGEmotions-27\eeg_raw",
                        help='Path to eeg_raw folder')
    parser.add_argument('--output', default='emotions_27.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    build_emotions27_csv(args.input, args.output)