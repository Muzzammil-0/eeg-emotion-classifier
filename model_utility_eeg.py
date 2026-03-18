import joblib
import numpy as np
import os
import glob


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