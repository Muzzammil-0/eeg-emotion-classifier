import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy import signal
import os
import glob

# ==============================================
# EMOTION MAPPING: 27 → 3 CLASSES
# ==============================================

EMOTION_MAP = {
    1:  'POSITIVE',   # admiration
    2:  'POSITIVE',   # adoration
    3:  'POSITIVE',   # aesthetic
    4:  'POSITIVE',   # amusement
    5:  'NEGATIVE',   # anger
    6:  'NEGATIVE',   # anxiety
    7:  'POSITIVE',   # awe
    8:  'NEUTRAL',    # awkwardness
    9:  'NEUTRAL',    # boredom
    10: 'NEUTRAL',    # calmness
    11: 'NEUTRAL',    # confusion
    12: 'NEUTRAL',    # craving
    13: 'NEGATIVE',   # disgust
    14: 'NEGATIVE',   # empathic pain
    15: 'POSITIVE',   # entrancement
    16: 'POSITIVE',   # excitement
    17: 'NEGATIVE',   # fear
    18: 'NEGATIVE',   # horror
    19: 'NEUTRAL',    # interest
    20: 'POSITIVE',   # joy
    21: 'POSITIVE',   # nostalgia
    22: 'POSITIVE',   # relief
    23: 'POSITIVE',   # romance
    24: 'NEGATIVE',   # sadness
    25: 'POSITIVE',   # satisfaction
    26: 'NEUTRAL',    # sexual desire
    27: 'POSITIVE',   # surprised
}

# Emotiv X 14 channel order (as they appear in the txt files)
EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                   'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# Channel indices matching dipps.py emotiv_x_14 registry mapping
TP9_IDX  = [EMOTIV_CHANNELS.index('P7'),  EMOTIV_CHANNELS.index('T7')]
AF7_IDX  = [EMOTIV_CHANNELS.index('AF3'), EMOTIV_CHANNELS.index('F7')]
AF8_IDX  = [EMOTIV_CHANNELS.index('AF4'), EMOTIV_CHANNELS.index('F8')]
TP10_IDX = [EMOTIV_CHANNELS.index('P8'),  EMOTIV_CHANNELS.index('T8')]

SOURCE_FS   = 256
TARGET_FS   = 150
TARGET_BINS = 500


def reduce_eeg_noise(eeg_signal, sfreq=150, notch_freq=50, lowcut=0.5, highcut=45):
    filtered = eeg_signal.copy()
    nyq = 0.5 * sfreq
    b_notch, a_notch = signal.iirnotch(notch_freq / nyq, Q=30)
    b_band,  a_band  = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    for ch in range(filtered.shape[0]):
        filtered[ch] = signal.filtfilt(b_notch, a_notch, filtered[ch])
        filtered[ch] = signal.filtfilt(b_band,  a_band,  filtered[ch])
    return filtered


def resample_signal(sig, source_fs, target_fs):
    n_samples = int(len(sig) * target_fs / source_fs)
    return signal.resample(sig, n_samples)


def pad_or_truncate(arr, target=500):
    if len(arr) >= target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)))


def load_txt(txt_path):
    """Try multiple encodings to load a txt file robustly."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return np.loadtxt(txt_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  SKIP {os.path.basename(txt_path)}: {e}")
            return None
    print(f"  SKIP {os.path.basename(txt_path)}: all encodings failed")
    return None


def process_txt_file(txt_path):
    data = load_txt(txt_path)
    if data is None:
        return None

    if data.ndim == 1:
        print(f"  SKIP {os.path.basename(txt_path)}: only 1 row")
        return None

    if data.shape[1] != 14:
        print(f"  SKIP {os.path.basename(txt_path)}: expected 14 cols, got {data.shape[1]}")
        return None

    data = data.T  # shape: (14, n_samples)

    # Resample 256 Hz → 150 Hz
    resampled = np.array([resample_signal(data[ch], SOURCE_FS, TARGET_FS)
                          for ch in range(14)])

    # Filter — same pipeline as dipps.py
    filtered = reduce_eeg_noise(resampled, sfreq=TARGET_FS)

    # Map to 4 Muse-equivalent channels using emotiv_x_14 registry logic
    tp9  = filtered[TP9_IDX].mean(axis=0)
    af7  = filtered[AF7_IDX].mean(axis=0)
    af8  = filtered[AF8_IDX].mean(axis=0)
    tp10 = filtered[TP10_IDX].mean(axis=0)

    combined = np.mean([tp9, af7, af8, tp10], axis=0)

    # FFT — same as dipps.py inference pipeline
    fft_vals = np.abs(rfft(combined))
    freqs    = rfftfreq(len(combined), 1 / TARGET_FS)
    max_bin  = np.searchsorted(freqs, 50.0)
    fft_500  = pad_or_truncate(fft_vals[:max_bin], TARGET_BINS)

    # Build row in Bird CSV format — duplicate into _a and _b
    row = {}
    for i in range(TARGET_BINS):
        row[f'fft_{i}_a'] = fft_500[i]
        row[f'fft_{i}_b'] = fft_500[i]

    return row


def preprocess_eeg27(input_dir, output_csv):
    txt_files = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    total     = len(txt_files)
    print(f"Found {total} txt files in {input_dir}\n")

    rows    = []
    skipped = 0

    for idx, txt_path in enumerate(txt_files):
        fname  = os.path.basename(txt_path)
        parts  = fname.replace('.txt', '').split('_')

        if len(parts) < 2:
            print(f"[{idx+1}/{total}] SKIP {fname}: unexpected filename")
            skipped += 1
            continue

        try:
            subject_id = int(parts[0])
            emotion_id = int(float(parts[1]))
        except ValueError:
            print(f"[{idx+1}/{total}] SKIP {fname}: cannot parse subject/emotion id")
            skipped += 1
            continue

        if emotion_id not in EMOTION_MAP:
            print(f"[{idx+1}/{total}] SKIP {fname}: unknown emotion id {emotion_id}")
            skipped += 1
            continue

        label = EMOTION_MAP[emotion_id]
        print(f"[{idx+1}/{total}] {fname} → subject {subject_id}, emotion {emotion_id} → {label}")

        row = process_txt_file(txt_path)
        if row is None:
            skipped += 1
            continue

        row['subject_id'] = f'eeg27_P{subject_id:03d}'
        row['label']      = label
        rows.append(row)

    df = pd.DataFrame(rows)

    # Column order matching Bird CSV format + subject_id
    fft_a_cols = [f'fft_{i}_a' for i in range(TARGET_BINS)]
    fft_b_cols = [f'fft_{i}_b' for i in range(TARGET_BINS)]
    df = df[fft_a_cols + fft_b_cols + ['subject_id', 'label']]

    df.to_csv(output_csv, index=False)

    print(f"\nDone.")
    print(f"Saved: {len(df)} samples → {output_csv}")
    print(f"Skipped: {skipped}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSubject count: {df['subject_id'].nunique()}")
    return df


if __name__ == '__main__':
    INPUT_DIR  = r'C:\Users\hp\Documents\EEGEmotions-27\eeg_raw'
    OUTPUT_CSV = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\emotions_27.csv'

    txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.txt')))
    total     = len(txt_files)
    print(f"Found {total} txt files\n")

    rows    = []
    skipped = 0

    for idx, txt_path in enumerate(txt_files):
        fname = os.path.basename(txt_path)
        parts = fname.replace('.txt', '').split('_')

        try:
            subject_id = int(parts[0])
            emotion_id = int(float(parts[1]))
        except (ValueError, IndexError):
            skipped += 1
            continue

        if emotion_id not in EMOTION_MAP:
            skipped += 1
            continue

        label = EMOTION_MAP[emotion_id]

        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{total}] processing... ({len(rows)} rows collected)")

        row = process_txt_file(txt_path)
        if row is None:
            skipped += 1
            continue

        row['subject_id'] = f'eeg27_P{subject_id:03d}'
        row['label']      = label
        rows.append(row)
        print(f"\nLoop done. {len(rows)} rows collected, {skipped} skipped.")
    print("Building DataFrame...")

    fft_a_cols = [f'fft_{i}_a' for i in range(TARGET_BINS)]
    fft_b_cols = [f'fft_{i}_b' for i in range(TARGET_BINS)]
    df = pd.DataFrame(rows)
    df = df[fft_a_cols + fft_b_cols + ['subject_id', 'label']]

    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Done. {df.shape}")
    print(df['label'].value_counts())
    print(f"Subjects: {df['subject_id'].nunique()}")
 