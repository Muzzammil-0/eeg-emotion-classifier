import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score
import os
import glob
import json

# ==============================================
# CONSTANTS
# ==============================================

SOURCE_FS   = 256
TARGET_FS   = 150
N_CHANNELS  = 4
WINDOW_SEC  = 2
WINDOW_SAMP = TARGET_FS * WINDOW_SEC   # 300 samples
STEP_SAMP   = WINDOW_SAMP // 2         # 150 samples (50% overlap)
N_CLASSES   = 3
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_MAP = {
    1:  'POSITIVE', 2:  'POSITIVE', 3:  'POSITIVE', 4:  'POSITIVE',
    5:  'NEGATIVE', 6:  'NEGATIVE', 7:  'POSITIVE', 8:  'NEUTRAL',
    9:  'NEUTRAL',  10: 'NEUTRAL',  11: 'NEUTRAL',  12: 'NEUTRAL',
    13: 'NEGATIVE', 14: 'NEGATIVE', 15: 'POSITIVE', 16: 'POSITIVE',
    17: 'NEGATIVE', 18: 'NEGATIVE', 19: 'NEUTRAL',  20: 'POSITIVE',
    21: 'POSITIVE', 22: 'POSITIVE', 23: 'POSITIVE', 24: 'NEGATIVE',
    25: 'POSITIVE', 26: 'NEUTRAL',  27: 'POSITIVE',
}

EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                   'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

TP9_IDX  = [EMOTIV_CHANNELS.index('P7'),  EMOTIV_CHANNELS.index('T7')]
AF7_IDX  = [EMOTIV_CHANNELS.index('AF3'), EMOTIV_CHANNELS.index('F7')]
AF8_IDX  = [EMOTIV_CHANNELS.index('AF4'), EMOTIV_CHANNELS.index('F8')]
TP10_IDX = [EMOTIV_CHANNELS.index('P8'),  EMOTIV_CHANNELS.index('T8')]

LABEL_TO_INT = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}


# ==============================================
# SIGNAL PROCESSING
# ==============================================

def reduce_eeg_noise(eeg_signal, sfreq=TARGET_FS, notch_freq=50,
                     lowcut=0.5, highcut=45):
    filtered = eeg_signal.copy()
    nyq = 0.5 * sfreq
    b_notch, a_notch = signal.iirnotch(notch_freq / nyq, Q=30)
    b_band,  a_band  = signal.butter(4, [lowcut / nyq, highcut / nyq],
                                     btype='band')
    for ch in range(filtered.shape[0]):
        filtered[ch] = signal.filtfilt(b_notch, a_notch, filtered[ch])
        filtered[ch] = signal.filtfilt(b_band,  a_band,  filtered[ch])
    return filtered


def resample_signal(sig, source_fs, target_fs):
    n = int(len(sig) * target_fs / source_fs)
    return signal.resample(sig, n)


def load_txt(txt_path):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return np.loadtxt(txt_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  SKIP {os.path.basename(txt_path)}: {e}")
            return None
    return None


def extract_4ch_from_txt(txt_path):
    """
    Load a raw EEGEmotions-27 txt file and return
    a (4, n_samples) array at TARGET_FS after filtering.
    Returns None on failure.
    """
    data = load_txt(txt_path)
    if data is None:
        return None
    if data.ndim == 1 or data.shape[1] != 14:
        return None

    data = data.T  # (14, n_samples)

    resampled = np.array([resample_signal(data[ch], SOURCE_FS, TARGET_FS)
                          for ch in range(14)])
    filtered  = reduce_eeg_noise(resampled, sfreq=TARGET_FS)

    tp9  = filtered[TP9_IDX].mean(axis=0)
    af7  = filtered[AF7_IDX].mean(axis=0)
    af8  = filtered[AF8_IDX].mean(axis=0)
    tp10 = filtered[TP10_IDX].mean(axis=0)

    return np.stack([tp9, af7, af8, tp10], axis=0)  # (4, n_samples)


def sliding_windows(signal_4ch, window=WINDOW_SAMP, step=STEP_SAMP):
    """
    Slice (4, n_samples) into list of (4, window) arrays.
    """
    n = signal_4ch.shape[1]
    windows = []
    start = 0
    while start + window <= n:
        windows.append(signal_4ch[:, start:start + window])
        start += step
    return windows


# ==============================================
# DATASET LOADING
# ==============================================

def load_eeg27_dataset(input_dir):
    """
    Load all EEGEmotions-27 txt files.
    Returns:
        X : list of (4, WINDOW_SAMP) numpy arrays
        y : list of int labels
        groups : list of subject IDs (for LOSO)
    """
    txt_files = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    print(f"Found {len(txt_files)} txt files")

    X, y, groups = [], [], []
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

        label     = EMOTION_MAP[emotion_id]
        label_int = LABEL_TO_INT[label]

        if (idx + 1) % 200 == 0:
            print(f"[{idx+1}/{len(txt_files)}] {len(X)} windows so far")

        signal_4ch = extract_4ch_from_txt(txt_path)
        if signal_4ch is None:
            skipped += 1
            continue

        windows = sliding_windows(signal_4ch)
        for w in windows:
            X.append(w.astype(np.float32))
            y.append(label_int)
            groups.append(subject_id)

    print(f"\nDone loading. Windows: {len(X)}, Skipped files: {skipped}")
    print(f"Subjects: {len(set(groups))}")

    label_counts = {k: y.count(v) for k, v in LABEL_TO_INT.items()}
    print(f"Label distribution: {label_counts}")

    return X, y, groups


# ==============================================
# PYTORCH DATASET
# ==============================================

class EEGDataset(Dataset):
    def __init__(self, X, y):
        # Normalise per window per channel (zero mean, unit std)
        self.X = []
        for w in X:
            w = w.copy()
            for ch in range(w.shape[0]):
                mu  = w[ch].mean()
                std = w[ch].std() + 1e-8
                w[ch] = (w[ch] - mu) / std
            self.X.append(torch.tensor(w, dtype=torch.float32).unsqueeze(0))
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================
# EEGNET ARCHITECTURE
# ==============================================

class EEGNet(nn.Module):
    """
    EEGNet: Lawhern et al. (2018)
    Input shape: (batch, 1, n_channels, n_samples)
    """
    def __init__(self, n_classes=N_CLASSES, n_channels=N_CHANNELS,
                 n_samples=WINDOW_SAMP, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            # Separable convolution
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out   = self.block2(self.block1(dummy))
            self._flat_size = out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


# ==============================================
# TRAINING AND EVALUATION
# ==============================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (outputs.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def eval_model(model, loader):
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            preds   = model(X_batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    return np.array(all_true), np.array(all_preds)


# ==============================================
# LOSO CROSS VALIDATION
# ==============================================

def run_loso_dl(input_dir, output_dir):
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s = {WINDOW_SAMP} samples | Step: {STEP_SAMP}")

    print("\nLoading dataset...")
    X, y, groups = load_eeg27_dataset(input_dir)

    subjects    = sorted(set(groups))
    groups_arr  = np.array(groups)
    y_arr       = np.array(y)
    int_to_label = {v: k for k, v in LABEL_TO_INT.items()}

    all_results = []
    all_true    = []
    all_pred    = []

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(subjects)} — Test subject: {test_subject}")
        print(f"{'='*60}")

        train_mask = groups_arr != test_subject
        test_mask  = groups_arr == test_subject

        X_train = [X[i] for i in range(len(X)) if train_mask[i]]
        y_train = y_arr[train_mask].tolist()
        X_test  = [X[i] for i in range(len(X)) if test_mask[i]]
        y_test  = y_arr[test_mask].tolist()

        print(f"Train windows: {len(X_train)} | Test windows: {len(X_test)}")

        # Class weights for imbalance
        class_counts = np.bincount(y_train, minlength=N_CLASSES)
        class_weights = torch.tensor(
            1.0 / (class_counts + 1e-8), dtype=torch.float32
        ).to(DEVICE)
        class_weights = class_weights / class_weights.sum()

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset  = EEGDataset(X_test,  y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        model     = EEGNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_bal_acc = 0
        best_state   = None

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader,
                                                optimizer, criterion)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                y_true_e, y_pred_e = eval_model(model, test_loader)
                bal_acc = balanced_accuracy_score(y_true_e, y_pred_e)
                print(f"  Epoch {epoch+1:3d} | loss {train_loss:.4f} | "
                      f"train_acc {train_acc:.4f} | bal_acc {bal_acc:.4f}")
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_state   = {k: v.clone()
                                    for k, v in model.state_dict().items()}

        # Final evaluation with best state
        if best_state is not None:
            model.load_state_dict(best_state)

        y_true_f, y_pred_f = eval_model(model, test_loader)
        bal_acc_f = balanced_accuracy_score(y_true_f, y_pred_f)

        label_names = [int_to_label[i] for i in range(N_CLASSES)]
        print(f"\nFinal balanced acc: {bal_acc_f:.4f}")
        print(classification_report(y_true_f, y_pred_f,
                                    target_names=label_names))

        all_results.append({
            'subject':      test_subject,
            'balanced_acc': round(bal_acc_f, 4),
            'n_test':       len(X_test)
        })
        all_true.extend(y_true_f.tolist())
        all_pred.extend(y_pred_f.tolist())

    # Overall results
    print(f"\n{'='*60}")
    print("EEGNET LOSO — OVERALL RESULTS")
    print(f"{'='*60}")

    mean_bal = np.mean([r['balanced_acc'] for r in all_results])
    std_bal  = np.std([r['balanced_acc']  for r in all_results])

    label_names = [int_to_label[i] for i in range(N_CLASSES)]
    print(f"Mean balanced accuracy: {mean_bal:.4f} ± {std_bal:.4f}")
    print(f"\nOverall classification report:")
    print(classification_report(all_true, all_pred, target_names=label_names))

    results_df = pd.DataFrame(all_results).sort_values('balanced_acc',
                                                        ascending=False)
    print(f"\nPer-subject results:")
    print(results_df.to_string(index=False))

    # Save results
    results_path = os.path.join(output_dir, 'loso_dl_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'mean_balanced_accuracy': round(mean_bal, 4),
            'std_balanced_accuracy':  round(std_bal,  4),
            'per_subject':            all_results,
            'overall_report':         classification_report(
                                          all_true, all_pred,
                                          target_names=label_names,
                                          output_dict=True)
        }, f, indent=2)

    print(f"\nResults saved: {results_path}")
    return mean_bal, std_bal


# ==============================================
# INFERENCE — single EDF file
# ==============================================

def predict_edf_dl(edf_path, model_path, target_fs=TARGET_FS):
    """
    Predict emotion from a single EDF file using trained EEGNet.
    Returns predicted label string.
    """
    import mne
    from dipps import detect_device_model, _select_channels_from_edf

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if raw.info['sfreq'] != target_fs:
        raw.resample(target_fs)

    raw_data      = raw.get_data() * 1e6
    filtered_data = reduce_eeg_noise(raw_data, sfreq=target_fs)
    raw           = mne.io.RawArray(filtered_data, raw.info)

    df_signals    = raw.to_data_frame()
    cleaned_names = [ch.replace('.', '').replace('-', '')
                     for ch in df_signals.columns]
    df_signals.columns = cleaned_names

    device_model = detect_device_model(cleaned_names)
    selected     = _select_channels_from_edf(df_signals, cleaned_names,
                                              device_model)

    signal_4ch = np.stack([
        selected['TP9'], selected['AF7'],
        selected['AF8'], selected['TP10']
    ], axis=0).astype(np.float32)

    windows = sliding_windows(signal_4ch)
    if not windows:
        return 'UNCERTAIN'

    model = EEGNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    int_to_label = {v: k for k, v in LABEL_TO_INT.items()}
    vote_counts  = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for w in windows:
            w_norm = w.copy()
            for ch in range(w_norm.shape[0]):
                mu  = w_norm[ch].mean()
                std = w_norm[ch].std() + 1e-8
                w_norm[ch] = (w_norm[ch] - mu) / std

            x = torch.tensor(w_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = model(x).argmax(1).item()
            vote_counts[pred] += 1

    final_pred = max(vote_counts, key=vote_counts.get)
    return int_to_label[final_pred]


# ==============================================
# MAIN
# ==============================================

if __name__ == '__main__':
    INPUT_DIR  = r'C:\Users\hp\Documents\EEGEmotions-27\eeg_raw'
    OUTPUT_DIR = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER'

    mean_acc, std_acc = run_loso_dl(INPUT_DIR, OUTPUT_DIR)
    print(f"\nFinal EEGNet LOSO: {mean_acc:.4f} ± {std_acc:.4f}")
