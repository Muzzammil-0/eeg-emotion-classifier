import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from collections import OrderedDict
import mne
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from scipy import signal
import os
import json

# ==============================================
# LRU CACHE FOR CHANNEL MAPPINGS
# ==============================================

_CHANNEL_CACHE = OrderedDict()
_MAX_CACHE_SIZE = 5


# ==============================================
# CORE SIGNAL PROCESSING FUNCTIONS
# ==============================================

def reduce_eeg_noise(eeg_signal, sfreq=150, notch_freq=50, lowcut=0.5, highcut=45):
    """Apply notch filter (remove powerline noise) and bandpass filter to EEG signal."""
    filtered = eeg_signal.copy()
    nyq = 0.5 * sfreq
    notch = notch_freq / nyq
    b_notch, a_notch = signal.iirnotch(notch, Q=30)
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = signal.butter(4, [low, high], btype='band')
    for ch in range(filtered.shape[0]):
        filtered[ch] = signal.filtfilt(b_notch, a_notch, filtered[ch])
        filtered[ch] = signal.filtfilt(b_band, a_band, filtered[ch])
    return filtered


def bins_to_waves(df_bins):
    """
    Convert 1002 FFT bin columns (fft_i_a, fft_i_b) to brainwave band features.

    The _a columns represent the left-hemisphere (or male-subject) signal.
    The _b columns represent the right-hemisphere (or female-subject) signal.
    Asymmetry = _a - _b, which is non-zero when the two sides differ.

    NOTE: Bird's dataset uses _a for the male subject and _b for the female subject.
    During inference we construct two rows — one with the new signal as _a against
    the female baseline as _b, and one reversed — then average both predictions.
    This is why asymmetry is meaningful even with a single input signal.
    """
    band_ranges = {
        'delta': (5,   40),
        'theta': (40,  80),
        'alpha': (80,  130),
        'beta':  (130, 300),
        'gamma': (300, 500)
    }

    wave_features = {}

    for band_name, (low, high) in band_ranges.items():
        a_cols = [f'fft_{i}_a' for i in range(low, high) if f'fft_{i}_a' in df_bins.columns]
        if a_cols:
            wave_features[f'{band_name}_A'] = df_bins[a_cols].mean(axis=1)

        b_cols = [f'fft_{i}_b' for i in range(low, high) if f'fft_{i}_b' in df_bins.columns]
        if b_cols:
            wave_features[f'{band_name}_B'] = df_bins[b_cols].mean(axis=1)

    for band in band_ranges.keys():
        a_key = f'{band}_A'
        b_key = f'{band}_B'
        if a_key in wave_features and b_key in wave_features:
            wave_features[f'{band}_asymmetry'] = wave_features[a_key] - wave_features[b_key]
            wave_features[f'{band}_symmetry']  = (wave_features[a_key] + wave_features[b_key]) / 2

    if 'alpha_A' in wave_features and 'alpha_B' in wave_features:
        wave_features['alpha_ratio'] = wave_features['alpha_A'] / (wave_features['alpha_B'] + 1e-10)
    if 'theta_A' in wave_features and 'beta_A' in wave_features:
        wave_features['theta_beta_ratio_A'] = wave_features['theta_A'] / (wave_features['beta_A'] + 1e-10)
    if 'theta_B' in wave_features and 'beta_B' in wave_features:
        wave_features['theta_beta_ratio_B'] = wave_features['theta_B'] / (wave_features['beta_B'] + 1e-10)

    return pd.DataFrame(wave_features)


def create_gender_baselines(x_train, train_targets, train_df):
    """
    Compute mean FFT profiles from the training data.

    In Bird's dataset, _a columns come from the male subject and _b from the female.
    These baselines are used at inference time to construct a comparison row for any
    new signal that has no known gender label.
    """
    a_cols = [f'fft_{i}_a' for i in range(500) if f'fft_{i}_a' in x_train.columns]
    b_cols = [f'fft_{i}_b' for i in range(500) if f'fft_{i}_b' in x_train.columns]

    if a_cols and b_cols:
        male_baseline   = x_train[a_cols].mean(axis=0).values
        female_baseline = x_train[b_cols].mean(axis=0).values
        return male_baseline, female_baseline
    else:
        all_cols = a_cols + b_cols
        all_mean = x_train[all_cols].mean(axis=0).values
        return all_mean, all_mean


def standardize_fft_bins(raw_fft, source_bins, target_bins=500):
    """Resize FFT array to exactly target_bins by padding or downsampling."""
    if source_bins == target_bins:
        return raw_fft
    elif source_bins < target_bins:
        return np.pad(raw_fft, (0, target_bins - source_bins), 'constant')
    else:
        if source_bins / target_bins > 2:
            ratio = source_bins // target_bins
            return np.mean(raw_fft[:target_bins * ratio].reshape(-1, ratio), axis=1)
        else:
            return raw_fft[:target_bins]


def _pad_or_truncate(arr, target=500):
    """Pad or truncate a 1D numpy array to exactly `target` elements."""
    if len(arr) >= target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)))


# ==============================================
# CHANNEL DETECTION AND MAPPING
# ==============================================

class EEGChannelMapper:
    """Maps arbitrary EEG channel names to the four Muse canonical positions."""

    def __init__(self):
        self.exact_matches = {
            'P7':  'TP9',  'P9':  'TP9',  'T7': 'TP9',  'T5': 'TP9',
            'P8':  'TP10', 'P10': 'TP10', 'T8': 'TP10', 'T6': 'TP10',
            'F7':  'AF7',  'AF7': 'AF7',  'F5': 'AF7',
            'F8':  'AF8',  'AF8': 'AF8',  'F6': 'AF8',
        }

    def find_muse_channel(self, eeg_channel_name):
        if eeg_channel_name in self.exact_matches:
            return self.exact_matches[eeg_channel_name]
        if eeg_channel_name.startswith(('P', 'T')) and any(x in eeg_channel_name for x in ['7', '5', '3']):
            return 'TP9' if '7' in eeg_channel_name or '5' in eeg_channel_name else None
        if eeg_channel_name.startswith(('P', 'T')) and any(x in eeg_channel_name for x in ['8', '6', '4']):
            return 'TP10' if '8' in eeg_channel_name or '6' in eeg_channel_name else None
        if eeg_channel_name.startswith('F') and any(x in eeg_channel_name for x in ['7', '5', '3', '1']):
            return 'AF7'
        if eeg_channel_name.startswith('F') and any(x in eeg_channel_name for x in ['8', '6', '4', '2']):
            return 'AF8'
        return None


def detect_device_model(channel_names):
    """Identify EEG device brand/model from channel name patterns."""
    cleaned     = [ch.replace('.', '').replace('-', '') for ch in channel_names]
    channel_set = set(cleaned)

    if 'TP9' in channel_set and 'AF7' in channel_set:
        return 'muse_4channel'
    if 'P9' in channel_set and 'P10' in channel_set:
        return 'natus_quantum_ltm'
    if ('F7' in channel_set and 'F8' in channel_set
            and 'P7' in channel_set and 'P8' in channel_set
            and 'P9' not in channel_set and 'P10' not in channel_set):
        return 'natus_emu40ex'

    physionet_patterns = ['Fc5', 'C5', 'Cp5']
    if any(p in str(channel_set) for p in physionet_patterns):
        return 'physionet_64_motor'

    biosemi_patterns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    if any(p in channel_names for p in biosemi_patterns):
        return 'biosemi_active2' if ('A32' in channel_names and 'B32' in channel_names) else 'biosemi_32'

    biosemi_std = ['Fp1', 'AF3', 'AF4', 'Fp2', 'PO3', 'PO4']
    if all(n in channel_names for n in biosemi_std[:3]):
        return 'biosemi_32'

    if 'A1' in channel_names and 'A2' in channel_names and len(channel_names) <= 25:
        if any(ch.startswith('T') for ch in channel_names):
            return 'nihon_kohden_25'

    if ('CP1' in channel_names and 'CP2' in channel_names and 'PO3' in channel_names
            and ('F5' in channel_names or 'F6' in channel_names)):
        return 'neuroscan_64'

    if ('AF7' in channel_names and 'AF8' in channel_names and 'PO7' in channel_names
            and 'FT7' in channel_names and 'FT8' in channel_names):
        return 'brainproducts_actichamp_64'

    if 'TP9' in channel_names and 'TP10' in channel_names:
        if 'FC1' in channel_names and 'FC2' in channel_names:
            return 'gtec_32'

    egi_pattern = any(ch.startswith('E') and ch[1:].isdigit() for ch in channel_names)
    if egi_pattern:
        max_num = max(
            (int(ch[1:]) for ch in channel_names if ch.startswith('E') and ch[1:].isdigit()),
            default=0
        )
        if max_num >= 128:
            return 'egi_128'
        elif max_num >= 64:
            return 'egi_64'
        else:
            return 'egi_32'
        
    emotiv_pattern = {'AF3', 'FC5', 'FC6', 'AF4'}
    if emotiv_pattern.issubset(channel_set):
        return 'emotiv_x_14'

    return 'generic_10_20'


# ==============================================
# MASTER CHANNEL REGISTRY (ALL DEVICES)
# ==============================================

CHANNEL_REGISTRY = {
    'muse_4channel': {
        'brand': 'muse', 'model': '2016',
        'channels': ['TP9', 'AF7', 'AF8', 'TP10', 'FPz'],
        'mapping': {'TP9': ['TP9'], 'AF7': ['AF7'], 'AF8': ['AF8'], 'TP10': ['TP10']}
    },
    'natus_quantum_ltm': {
        'brand': 'natus', 'model': 'quantum_ltm',
        'channels': ['P9', 'P10', 'AF3', 'AF4', 'F7', 'F8', 'P7', 'P8'],
        'mapping': {
            'TP9':  ['P9',  'P7', 'T5', 'M1', 'CB1'],
            'AF7':  ['AF3', 'F7', 'F5', 'F3', 'Fp1'],
            'AF8':  ['AF4', 'F8', 'F6', 'F4', 'Fp2'],
            'TP10': ['P10', 'P8', 'T6', 'M2', 'CB2']
        }
    },
    'natus_emu40ex': {
        'brand': 'natus', 'model': 'emu40ex',
        'channels': ['F7', 'F8', 'P7', 'P8', 'T7', 'T8'],
        'mapping': {
            'TP9':  ['P7', 'T5', 'P3'],
            'AF7':  ['F7', 'F3'],
            'AF8':  ['F8', 'F4'],
            'TP10': ['P8', 'T6', 'P4']
        }
    },
    'physionet_64_motor': {
        'brand': 'physionet', 'model': 'BCI2000_64',
        'channels': [
            'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
            'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
            'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.',
        ],
        'mapping': {
            'TP9':  ['Cp5.', 'Cp3.', 'C5..', 'C3..'],
            'AF7':  ['Fc5.', 'Fc3.', 'Fc1.'],
            'AF8':  ['Fc2.', 'Fc4.', 'Fc6.'],
            'TP10': ['Cp2.', 'Cp4.', 'C4..', 'C6..']
        }
    },
    'biosemi_active2': {
        'brand': 'biosemi', 'model': 'active2',
        'channels': [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)],
        'mapping': {
            'TP9':  ['B21', 'B22', 'B23'],
            'AF7':  ['A2',  'A3',  'A4'],
            'AF8':  ['B2',  'B3',  'B4'],
            'TP10': ['B28', 'B29', 'B30']
        }
    },
    'biosemi_32': {
        'brand': 'biosemi', 'model': '32_channel',
        'channels': [
            'Fp1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',
            'CP1', 'CP5', 'P7',  'P3',  'Pz',  'PO3', 'O1',  'Oz',
            'O2',  'PO4', 'P4',  'P8',  'CP6', 'CP2', 'C4',  'T8',
            'FC6', 'FC2', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz'
        ],
        'mapping': {
            'TP9':  ['P7',  'T7', 'CP5'],
            'AF7':  ['AF3', 'F7'],
            'AF8':  ['AF4', 'F8'],
            'TP10': ['P8',  'T8', 'CP6']
        }
    },
    'nihon_kohden_25': {
        'brand': 'nihon_kohden', 'model': '25_channel',
        'channels': [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7',  'F8',  'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2',
            'Fpz', 'Oz',  'Fcz', 'Cpz'
        ],
        'mapping': {
            'TP9':  ['T5', 'P3', 'A1'],
            'AF7':  ['F7', 'F3', 'Fp1'],
            'AF8':  ['F8', 'F4', 'Fp2'],
            'TP10': ['T6', 'P4', 'A2']
        }
    },
    'nihon_kohden_64': {
        'brand': 'nihon_kohden', 'model': '64_channel',
        'channels': [f'ch{i}' for i in range(1, 65)],
        'mapping': {
            'TP9':  ['ch17', 'ch18', 'ch19'],
            'AF7':  ['ch1',  'ch2',  'ch3'],
            'AF8':  ['ch62', 'ch63', 'ch64'],
            'TP10': ['ch46', 'ch47', 'ch48']
        }
    },
    'neuroscan_64': {
        'brand': 'compumedics_neuroscan', 'model': '64_channel',
        'channels': [
            'Fp1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',  'CP1', 'CP5', 'P7',  'P3',
            'Pz',  'PO3', 'O1',  'Oz',  'O2',  'PO4', 'P4',  'P8',  'CP6', 'CP2', 'C4',  'T8',
            'FC6', 'FC2', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz',  'CPz', 'POz', 'F1',  'F2',
            'C1',  'C2',  'P1',  'P2',  'AF7', 'AF8', 'F5',  'F6',  'C5',  'C6',  'P5',  'P6',
            'PO7', 'PO8', 'FT7', 'FT8', 'TP7', 'TP8', 'CP3', 'CP4', 'FC3', 'FC4', 'F9',  'F10',
            'P9',  'P10', 'Iz'
        ],
        'mapping': {
            'TP9':  ['P7',  'P9',  'TP7', 'T7',  'P3',  'CP5'],
            'AF7':  ['AF7', 'F7',  'F5',  'AF3', 'F3',  'Fp1'],
            'AF8':  ['AF8', 'F8',  'F6',  'AF4', 'F4',  'Fp2'],
            'TP10': ['P8',  'P10', 'TP8', 'T8',  'P4',  'CP6']
        }
    },
    'neuroscan_32': {
        'brand': 'compumedics_neuroscan', 'model': '32_channel',
        'channels': [
            'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',  'FC1', 'FC2', 'FC5', 'FC6',
            'T7',  'C3',  'Cz',  'C4',  'T8',  'CP1', 'CP2', 'CP5', 'CP6',
            'P7',  'P3',  'Pz',  'P4',  'P8',  'PO3', 'PO4', 'O1',  'Oz',  'O2',
            'A1',  'A2'
        ],
        'mapping': {
            'TP9':  ['P7',  'T7',  'CP5', 'A1'],
            'AF7':  ['F7',  'FC5', 'F3',  'Fp1'],
            'AF8':  ['F8',  'FC6', 'F4',  'Fp2'],
            'TP10': ['P8',  'T8',  'CP6', 'A2']
        }
    },
    'neuroscan_nuamps': {
        'brand': 'compumedics_neuroscan', 'model': 'nuamps',
        'channels': [
            'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',  'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
            'T7',  'C3',  'Cz',  'C4',  'T8',  'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
            'P7',  'P3',  'Pz',  'P4',  'P8',  'O1',  'Oz',  'O2',  'A1',  'A2'
        ],
        'mapping': {
            'TP9':  ['P7',  'T7',  'TP7', 'A1'],
            'AF7':  ['F7',  'FT7', 'FC3', 'Fp1'],
            'AF8':  ['F8',  'FT8', 'FC4', 'Fp2'],
            'TP10': ['P8',  'T8',  'TP8', 'A2']
        }
    },
    'neuroscan_synamps': {
        'brand': 'compumedics_neuroscan', 'model': 'synamps_rt',
        'channels': [f'ch{i}' for i in range(1, 65)],
        'mapping': {
            'TP9':  ['P7',  'T7',  'CP5'],
            'AF7':  ['F7',  'AF3', 'F3'],
            'AF8':  ['F8',  'AF4', 'F4'],
            'TP10': ['P8',  'T8',  'CP6']
        }
    },
    'brainproducts_actichamp_64': {
        'brand': 'brain_products', 'model': 'actichamp_64',
        'channels': [
            'Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',  'FT7', 'FC5', 'FC3', 'FC1',
            'C1',  'C3',  'C5',  'T7',  'TP7', 'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',
            'P7',  'P9',  'PO7', 'PO3', 'O1',  'Oz',  'POz', 'Pz',  'CPz', 'Fpz', 'Fp2',
            'AF8', 'AF4', 'F2',  'F4',  'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2', 'C2',
            'C4',  'C6',  'T8',  'TP8', 'CP6', 'CP4', 'CP2', 'P2',  'P4',  'P6',  'P8',
            'P10', 'PO8', 'PO4', 'O2',  'Cz',  'Fz',  'FCz', 'Iz',  'A1',  'A2'
        ],
        'mapping': {
            'TP9':  ['P7',  'P9',  'TP7', 'T7',  'P3',  'A1'],
            'AF7':  ['AF7', 'F7',  'F5',  'AF3', 'F3',  'Fp1'],
            'AF8':  ['AF8', 'F8',  'F6',  'AF4', 'F4',  'Fp2'],
            'TP10': ['P8',  'P10', 'TP8', 'T8',  'P4',  'A2']
        }
    },
    'brainproducts_actichamp_32': {
        'brand': 'brain_products', 'model': 'actichamp_32',
        'channels': [
            'Fp1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',  'CP1', 'CP5', 'P7',
            'P3',  'Pz',  'PO3', 'O1',  'Oz',  'O2',  'PO4', 'P4',  'P8',  'CP6', 'CP2',
            'C4',  'T8',  'FC6', 'FC2', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz'
        ],
        'mapping': {
            'TP9':  ['P7',  'T7',  'CP5', 'P3'],
            'AF7':  ['AF3', 'F7',  'FC5', 'F3',  'Fp1'],
            'AF8':  ['AF4', 'F8',  'FC6', 'F4',  'Fp2'],
            'TP10': ['P8',  'T8',  'CP6', 'P4']
        }
    },
    'gtec_32': {
        'brand': 'gtec', 'model': 'g.USBamp_32',
        'channels': [
            'F7',  'F8',  'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'Fz',  'C1',  'C2',  'C3',  'C4',
            'C5',  'C6',  'CP3', 'CP4', 'CP5', 'CP6', 'FT7', 'FT8', 'P3',  'P4',  'P5',  'P6',
            'P7',  'P8',  'T7',  'T8',  'TP7', 'TP8', 'TP9', 'TP10', 'Cz', 'CPz', 'Pz'
        ],
        'mapping': {
            'TP9':  ['TP9',  'P7',  'TP7', 'P3'],
            'AF7':  ['F7',   'FC5', 'FT7', 'FC3'],
            'AF8':  ['F8',   'FC6', 'FT8', 'FC4'],
            'TP10': ['TP10', 'P8',  'TP8', 'P4']
        }
    },
    'gtec_nautilus': {
        'brand': 'gtec', 'model': 'nautilus',
        'channels': [
            'Fp1', 'Fp2', 'Fz',  'F3',  'F4',  'F7',  'F8',  'Cz',  'C3',  'C4',
            'Pz',  'P3',  'P4',  'P7',  'P8',  'PO7', 'PO8', 'O1',  'O2',  'Oz',
            'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'T7',  'T8',
            'AF3', 'AF4', 'F1',  'F2',  'C1',  'C2',  'P1',  'P2',  'PO3', 'PO4'
        ],
        'mapping': {
            'TP9':  ['P7',  'P3',  'CP5', 'T7'],
            'AF7':  ['AF3', 'F7',  'F5',  'F3',  'Fp1'],
            'AF8':  ['AF4', 'F8',  'F6',  'F4',  'Fp2'],
            'TP10': ['P8',  'P4',  'CP6', 'T8']
        }
    },
    'gtec_sahara': {
        'brand': 'gtec', 'model': 'sahara',
        'channels': [
            'Fz',  'Cz',  'Pz',  'F3',  'F4',  'C3',  'C4',  'P3',  'P4',
            'FC1', 'FC2', 'CP1', 'CP2', 'F7',  'F8',  'T7',  'T8',  'P7',  'P8',
            'O1',  'O2',  'AF3', 'AF4', 'F5',  'F6',  'C5',  'C6',  'P5',  'P6',
            'FC5', 'FC6', 'CP5', 'CP6'
        ],
        'mapping': {
            'TP9':  ['P7',  'T7',  'CP5', 'P3'],
            'AF7':  ['AF3', 'F7',  'F5',  'F3'],
            'AF8':  ['AF4', 'F8',  'F6',  'F4'],
            'TP10': ['P8',  'T8',  'CP6', 'P4']
        }
    },
    'egi_128': {
        'brand': 'egi', 'model': 'ges_400_128',
        'channels': [f'E{i}' for i in range(1, 129)],
        'mapping': {
            'TP9':  ['E46', 'E47', 'E52', 'E53'],
            'AF7':  ['E22', 'E23', 'E27', 'E28'],
            'AF8':  ['E9',  'E15', 'E16', 'E10'],
            'TP10': ['E91', 'E96', 'E97', 'E101']
        }
    },
    'egi_64': {
        'brand': 'egi', 'model': 'ges_400_64',
        'channels': [f'E{i}' for i in range(1, 65)],
        'mapping': {
            'TP9':  ['E47', 'E52', 'E53', 'E46'],
            'AF7':  ['E22', 'E23', 'E27'],
            'AF8':  ['E9',  'E15', 'E16'],
            'TP10': ['E91', 'E96', 'E97']
        }
    },
    'egi_256': {
        'brand': 'egi', 'model': 'ges_400_256',
        'channels': [f'E{i}' for i in range(1, 257)],
        'mapping': {
            'TP9':  ['E46', 'E47', 'E52', 'E53', 'E58',  'E59'],
            'AF7':  ['E22', 'E23', 'E24', 'E27', 'E28',  'E33'],
            'AF8':  ['E1',  'E8',  'E9',  'E15', 'E16',  'E17'],
            'TP10': ['E91', 'E92', 'E96', 'E97', 'E101', 'E102']
        }
    },
    'egi_32': {
        'brand': 'egi', 'model': 'ges_400_32',
        'channels': [f'E{i}' for i in range(1, 33)],
        'mapping': {
            'TP9':  ['E17', 'E18', 'E19'],
            'AF7':  ['E22', 'E23', 'E24'],
            'AF8':  ['E1',  'E8',  'E9'],
            'TP10': ['E10', 'E11', 'E12']
        }
    },

    'emotiv_x_14': {
        'brand': 'emotiv', 'model': 'epoc_x',
        'channels': ['AF3', 'F7', 'FC5', 'T7', 'P7', '01',
                     '02', 'P8', 'T8', 'FC6', 'F4','F8', 'AF4'],
        'mapping': {
            'TP9': ['P7', 'T7', '01'],
            'AF7': ['AF3', 'F7', 'F3'],
            'AF8': ['AF4', 'F8', 'F4'],
            'TP10': ['P8', 'T8', '02']
        }
    },

    'generic_10_20': {
        'brand': 'unknown', 'model': '10-20_system',
        'channels': [],
        'mapping': {
            'TP9':  ['P7',   'P9',   'T7',   'T5',   'P3',   'Cp5.', 'Cp3.', 'C5..'],
            'AF7':  ['F7',   'AF7',  'F5',   'F3',   'Fc5.', 'Fc3.', 'Fc1.'],
            'AF8':  ['F8',   'AF8',  'F6',   'F4',   'Fc2.', 'Fc4.', 'Fc6.'],
            'TP10': ['P8',   'P10',  'T8',   'T6',   'P4',   'Cp2.', 'Cp4.', 'C4..']
        }
    }
}


# ==============================================
# EDF FEATURE EXTRACTION
# ==============================================

def _select_channels_from_edf(df_signals, cleaned_names, device_model):
    """
    Internal helper: maps EDF channel columns to the four canonical Muse positions.
    Returns a dict {muse_channel: np.array} or raises ValueError if a channel is missing.
    """
    mapping = (CHANNEL_REGISTRY[device_model]['mapping']
               if device_model in CHANNEL_REGISTRY
               else CHANNEL_REGISTRY['generic_10_20']['mapping'])

    mapper   = EEGChannelMapper()
    selected = {}

    for muse_ch, options in mapping.items():
        found = False
        for opt in options:
            opt_clean = opt.replace('.', '').replace('-', '')
            if opt_clean in cleaned_names:
                selected[muse_ch] = df_signals[opt_clean].values
                found = True
                break
        if not found:
            for ch in cleaned_names:
                if mapper.find_muse_channel(ch) == muse_ch:
                    selected[muse_ch] = df_signals[ch].values
                    found = True
                    break

    required = ['TP9', 'AF7', 'AF8', 'TP10']
    for ch in required:
        if ch not in selected:
            raise ValueError(f'Required channel {ch} could not be mapped from the EDF file')

    return selected


def extract_features_for_training(edf_path, target_fs=150):
    """
    Load an EDF file and extract wave-band features suitable for model training.
    Returns a 1D numpy array of features, or None on failure.
    """
    print(f"\n  Processing: {edf_path}")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        print(f"    Loaded: {len(raw.ch_names)} channels at {raw.info['sfreq']} Hz")

        raw_data      = raw.get_data() * 1e6  # convert V to µV
        filtered_data = reduce_eeg_noise(raw_data, sfreq=raw.info['sfreq'])
        raw           = mne.io.RawArray(filtered_data, raw.info)

        if raw.info['sfreq'] != target_fs:
            raw.resample(target_fs)

        df_signals    = raw.to_data_frame()
        cleaned_names = [ch.replace('.', '').replace('-', '') for ch in df_signals.columns]
        df_signals.columns = cleaned_names

        device_model = detect_device_model(cleaned_names)
        print(f"  Detected device: {device_model}")

        selected = _select_channels_from_edf(df_signals, cleaned_names, device_model)

        combined_signal = np.mean([selected['TP9'],  selected['AF7'],
                                   selected['AF8'],  selected['TP10']], axis=0)

        fft_vals = np.abs(rfft(combined_signal))
        freqs    = rfftfreq(len(combined_signal), 1 / target_fs)
        max_bin  = np.searchsorted(freqs, 50.0)
        fft_500  = _pad_or_truncate(fft_vals[:max_bin])

        row = {}
        for i in range(500):
            row[f'fft_{i}_a'] = fft_500[i]
            row[f'fft_{i}_b'] = fft_500[i]

        wave_features = bins_to_waves(pd.DataFrame([row])).iloc[0].values
        print("  Features extracted successfully")
        return wave_features

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def edf_to_csv(edf_path, csv_path, target_fs=150):
    """Convert an EDF recording to a CSV file at the target sampling rate."""
    print(f"Reading: {edf_path}")
    raw         = mne.io.read_raw_edf(edf_path, preload=True)
    original_fs = raw.info['sfreq']
    print(f"Original rate: {original_fs} Hz  |  Channels: {raw.ch_names}")
    if original_fs != target_fs:
        raw.filter(0.1, target_fs / 2 * 0.9, fir_design='firwin')
        raw.resample(target_fs)
    df = raw.to_data_frame()
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({df.shape[0]} samples x {df.shape[1]} channels)")
    return df


# ==============================================
# INFERENCE
# ==============================================

def predict_emotion_from_edf_single(edf_path, model, le, male_baseline, female_baseline,
                                     target_fs=150, scaler=None, verbose=False):
    """
    Predict emotion label from a single EDF file.

    Parameters
    ----------
    edf_path         : path to EDF file
    model            : trained VotingClassifier
    le               : fitted LabelEncoder
    male_baseline    : mean FFT profile from _a (male) training columns
    female_baseline  : mean FFT profile from _b (female) training columns
    target_fs        : target sampling rate (default 150 Hz)
    scaler           : fitted MinMaxScaler from training (recommended).
                       If None, raw FFT values are used — may reduce accuracy.
                       i will say it indeed has reduced accuracy by 2%
    verbose          : if True, prints per-branch probability vectors

    Strategy
    --------
    We don't know the gender of a new subject, so inference runs twice:
      1. Signal as _a (male side), paired with female baseline as _b.
      2. Signal as _b (female side), paired with male baseline as _a.
    Both probability vectors are averaged before taking argmax.
    This keeps asymmetry features meaningful for any input.
    """
    # --- Check cache ---
    if edf_path in _CHANNEL_CACHE:
        print("  Using cached channel mapping")
        selected = _CHANNEL_CACHE[edf_path]
        _CHANNEL_CACHE.move_to_end(edf_path)
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        if raw.info['sfreq'] != target_fs:
            raw.resample(target_fs)

        raw_data      = raw.get_data() * 1e6  # convert V to µV
        filtered_data = reduce_eeg_noise(raw_data, sfreq=target_fs)
        raw           = mne.io.RawArray(filtered_data, raw.info)

        df_signals    = raw.to_data_frame()
        cleaned_names = [ch.replace('.', '').replace('-', '') for ch in df_signals.columns]
        df_signals.columns = cleaned_names

        device_model = detect_device_model(cleaned_names)
        print(f"  Detected device: {device_model}")

        selected = _select_channels_from_edf(df_signals, cleaned_names, device_model)

        _CHANNEL_CACHE[edf_path] = selected
        if len(_CHANNEL_CACHE) > _MAX_CACHE_SIZE:
            _CHANNEL_CACHE.popitem(last=False)

    # --- Compute FFT on averaged signal ---
    combined_signal = np.mean([selected['TP9'],  selected['AF7'],
                               selected['AF8'],  selected['TP10']], axis=0)
    fft_vals = np.abs(rfft(combined_signal))
    freqs    = rfftfreq(len(combined_signal), 1 / target_fs)
    max_bin  = np.searchsorted(freqs, 50.0)
    fft_500  = _pad_or_truncate(fft_vals[:max_bin])

    # --- Scale FFT to match training distribution ---
    if scaler is not None:
        
        all_cols = [f'fft_{i}_a' for i in range(500)] + [f'fft_{i}_b' for i in range(500)]
        raw_row  = np.concatenate([fft_500, fft_500]).reshape(1, -1)
        scaled   = scaler.transform(pd.DataFrame(raw_row, columns=all_cols))[0]
        fft_a    = scaled[:500]
        fft_b    = scaled[500:1000]
    else:
        print("  WARNING: no scaler provided — using raw FFT values (may reduce accuracy)")
        fft_a = fft_500
        fft_b = fft_500

    # --- Dual-baseline inference ---
    row_male   = {f'fft_{i}_a': fft_a[i] for i in range(500)}
    row_male.update({f'fft_{i}_b': female_baseline[i] for i in range(500)})

    row_female = {f'fft_{i}_b': fft_b[i] for i in range(500)}
    row_female.update({f'fft_{i}_a': male_baseline[i] for i in range(500)})

    features_male   = bins_to_waves(pd.DataFrame([row_male]))
    features_female = bins_to_waves(pd.DataFrame([row_female]))

    proba_male   = model.predict_proba(features_male)[0]
    proba_female = model.predict_proba(features_female)[0]
    avg_proba    = (proba_male + proba_female) / 2

    if verbose:
        print(f"  proba_male:   {proba_male}")
        print(f"  proba_female: {proba_female}")
        print(f"  avg_proba:    {avg_proba}")
        print(f"  classes:      {le.classes_}")

    pred_encoded = np.argmax(avg_proba)
    return le.inverse_transform([pred_encoded])[0]


# ==============================================
# TRAINING
# ==============================================

if __name__ == "__main__":
    import argparse
    import sklearn
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    parser = argparse.ArgumentParser(description='Train EEG emotion classifier')
    parser.add_argument('--data',    default=r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\emotions_combined.csv',
                        help='Path to training CSV (default: emotions.csv)')
    parser.add_argument('--version', default='version_0',
                        help='Version name for saved artefacts (default: version_0)')
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    EEG_df = pd.read_csv(args.data, skiprows=1, header=None)
    with open(args.data, 'r') as f:
        header_line  = f.readline().strip('# \n')
        column_names = header_line.split(',')
    EEG_df.columns = column_names

    print(f"Dataset shape: {EEG_df.shape}")
    print(f"Null counts:\n{EEG_df[column_names].isnull().sum()}")

    train_df, test_df = train_test_split(EEG_df, test_size=0.1, random_state=42)

    input_cols   = [f'fft_{i}_a' for i in range(500)] + [f'fft_{i}_b' for i in range(500)]
    target_col   = 'label'
    all_numeric  = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in all_numeric if col in input_cols]

    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols]  = imputer.transform(test_df[numeric_cols])

    scaler = MinMaxScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols]  = scaler.transform(test_df[numeric_cols])
    joblib.dump(scaler, f'scaler_{args.version}.pkl')
    print(f"Scaler saved: scaler_{args.version}.pkl")

    x_train       = train_df[numeric_cols]
    train_targets = train_df[target_col]
    x_test        = test_df[numeric_cols]
    test_targets  = test_df[target_col]

    x_train_waves = bins_to_waves(x_train)
    x_test_waves  = bins_to_waves(x_test)

    male_baseline, female_baseline = create_gender_baselines(x_train, train_targets, train_df)

    le                = LabelEncoder()
    train_targets_enc = le.fit_transform(train_targets)
    test_targets_enc  = le.transform(test_targets)

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=train_targets_enc
    )

    sklearn.set_config(enable_metadata_routing=True)

    dt  = DecisionTreeClassifier(random_state=42,
                                  class_weight='balanced').set_fit_request(sample_weight=True)
    rf  = RandomForestClassifier(random_state=42,
                                  class_weight='balanced').set_fit_request(sample_weight=True)
    xgb = XGBClassifier(random_state=42,
                         n_jobs=-1).set_fit_request(sample_weight=True)

    voting_clf = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 5]
    )

    param_grid = {
        'xgb__max_depth':        [2, 3, 4],
        'xgb__n_estimators':     [50, 100],
        'xgb__min_child_weight': [1, 5, 10],
        'rf__max_depth':         [3, 4],
        'rf__n_estimators':      [50, 100],
        'rf__min_samples_leaf':  [1, 5, 10],
        'dt__max_depth':         [2, 3],
        'dt__min_samples_leaf':  [1, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Searching for best hyperparameters...")
    grid_search = GridSearchCV(
        voting_clf,
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train_waves, train_targets_enc,
                    sample_weight=sample_weights)

    print(f"Best params:   {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    voting_clf = grid_search.best_estimator_

    train_acc = voting_clf.score(x_train_waves, train_targets_enc)
    test_acc  = voting_clf.score(x_test_waves,  test_targets_enc)
    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Test accuracy  : {test_acc:.4f}")

    y_pred = voting_clf.predict(x_test_waves)
    report = classification_report(test_targets_enc, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print(classification_report(test_targets_enc, y_pred, target_names=le.classes_))

    metrics_path = f'metrics_{args.version}.json'
    with open(metrics_path, 'w') as f:
        json.dump({'train_accuracy': train_acc, 'test_accuracy': test_acc,
                   'classification_report': report}, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    from model_utility_eeg import save_training_state
    save_training_state(
        version_name=args.version,
        X=x_train_waves,
        y=train_targets,
        model=voting_clf,
        le=le,
        male_baseline=male_baseline,
        female_baseline=female_baseline
    )

    np.save(f'X_test_{args.version}.npy',       x_test_waves)
    np.save(f'y_test_{args.version}.npy',        test_targets_enc)
    np.save(f'y_test_labels_{args.version}.npy', test_targets)

    print("Done.")