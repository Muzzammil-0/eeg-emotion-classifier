
import joblib
import numpy as np
import os

def save_training_state(version_0, X, y, model, le, male_baseline, female_baseline):
    """Save complete training state for rollback"""
    import joblib
    import numpy as np
    
    # Save data
    np.save(f'X_train_{version_0}.npy', X)
    np.save(f'y_train_{version_0}.npy', y)
    
    # Save model
    joblib.dump(model, f'model_{version_0}.pkl')
    joblib.dump(le, f'label_encoder_{version_0}.pkl')
    
    # Save baselines
    np.save(f'male_baseline_{version_0}.npy', male_baseline)
    np.save(f'female_baseline_{version_0}.npy', female_baseline)
    
    print(f" Saved training state: {version_0}")

def load_training_state(version_name):
    
    X = np.load(f'X_train_{version_name}.npy')
    y = np.load(f'y_train_{version_name}.npy', allow_pickle=True)
    model = joblib.load(f'model_{version_name}.pkl')
    le = joblib.load(f'label_encoder_{version_name}.pkl')
    male_baseline = np.load(f'male_baseline_{version_name}.npy')
    female_baseline = np.load(f'female_baseline_{version_name}.npy')
    
    print(f" Loaded version: {version_name}")
    print(f"   Features shape: {X.shape}")
    print(f"   Samples: {len(y)}")
    
    return X, y, model, le, male_baseline, female_baseline

def list_available_versions():
    """List all saved model versions"""
    import glob
    
    models = glob.glob('model_*.pkl')
    versions = [m.replace('model_', '').replace('.pkl', '') for m in models]
    
    print(" Available versions:")
    for v in versions:
        # Check if all corresponding files exist
        if (os.path.exists(f'X_train_{v}.npy') and 
            os.path.exists(f'y_train_{v}.npy') and
            os.path.exists(f'label_encoder_{v}.pkl')):
            print(f"   - {v}")
    
    return versions

