import joblib
import numpy as np
import os
import glob
import re

def save_training_state(version_name, X, y, model, le, male_baseline, female_baseline):
    """Save a complete training snapshot so it can be loaded or rolled back later."""
    np.save(f'X_train_{version_name}.npy',            X)
    np.save(f'y_train_{version_name}.npy',            y)
    joblib.dump(model, f'model_{version_name}.pkl')
    joblib.dump(le,    f'label_encoder_{version_name}.pkl')
    np.save(f'male_baseline_{version_name}.npy',   male_baseline)
    np.save(f'female_baseline_{version_name}.npy', female_baseline)
    print(f"  Saved training state: {version_name}")

def load_training_state(version_name):
    """Load a previously saved training snapshot."""
    X              = np.load(f'X_train_{version_name}.npy')
    y              = np.load(f'y_train_{version_name}.npy', allow_pickle=True)
    model          = joblib.load(f'model_{version_name}.pkl')
    le             = joblib.load(f'label_encoder_{version_name}.pkl')
    male_baseline  = np.load(f'male_baseline_{version_name}.npy')
    female_baseline= np.load(f'female_baseline_{version_name}.npy')
    print(f"  Loaded version : {version_name}")
    print(f"  Features shape : {X.shape}")
    print(f"  Samples        : {len(y)}")
    return X, y, model, le, male_baseline, female_baseline

def list_available_versions():
    """Print and return all complete saved model versions."""
    models   = glob.glob('model_*.pkl')
    versions = [m.replace('model_', '').replace('.pkl', '') for m in models]
    complete = []
    print("  Available versions:")
    for v in versions:
        if (os.path.exists(f'X_train_{v}.npy')
                and os.path.exists(f'y_train_{v}.npy')
                and os.path.exists(f'label_encoder_{v}.pkl')):
            print(f"    - {v}")
            complete.append(v)
    return complete

def cleanup_old_versions(keep_last=2):
    """
    Delete all versioned files except the most recent `keep_last` versions.
    Keeps only the latest two versions by default.
    Works with version names like 'version_0', 'version_1_trained', etc.
    """
    # Patterns for all versioned file types
    patterns = [
        'X_train_*.npy', 'y_train_*.npy', 'X_test_*.npy', 'y_test_*.npy', 'y_test_labels_*.npy',
        'model_*.pkl', 'label_encoder_*.pkl', 'male_baseline_*.npy', 'female_baseline_*.npy',
        'scaler_*.pkl', 'metrics_*.json'
    ]
    
    # Collect all files with version info
    version_files = {}
    version_pattern = re.compile(r'_(version_\d+(?:_trained)?)\.(?:pkl|npy|json)$')
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            match = version_pattern.search(filepath)
            if match:
                ver = match.group(1)  # e.g., "version_0" or "version_1_trained"
                version_files.setdefault(ver, []).append(filepath)
    
    if not version_files:
        return
    
    # Sort versions by numeric part (ignoring _trained for ordering)
    def version_key(v):
        base = v.replace('_trained', '')
        num = int(base.split('_')[1])
        return (num, 0 if '_trained' not in v else 1)
    
    sorted_versions = sorted(version_files.keys(), key=version_key, reverse=True)
    to_delete = sorted_versions[keep_last:]  # keep first 'keep_last'
    
    for ver in to_delete:
        for filepath in version_files[ver]:
            try:
                os.remove(filepath)
                print(f"  Deleted old version file: {filepath}")
            except Exception as e:
                print(f"  Error deleting {filepath}: {e}")
