# compare_versions.py
from model_utility_eeg import load_training_state
from sklearn.metrics import accuracy_score
import numpy as np
import os
import glob
import re

def compare_versions(version_a, version_b, test_X_path='X_test_version_0.npy', test_y_path='y_test_version_0.npy'):
    # Load fixed test set (saved during training)
    X_test = np.load(test_X_path)
    y_test = np.load(test_y_path)   

    # Load both model versions
    X_a, y_a, model_a, le_a, _, _ = load_training_state(version_a)
    X_b, y_b, model_b, le_b, _, _ = load_training_state(version_b)

    # Predict
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)

    acc_a = accuracy_score(y_test, pred_a)
    acc_b = accuracy_score(y_test, pred_b)

    print(f"\nComparing {version_a} vs {version_b}")
    print("-" * 40)
    print(f"{version_a}: {acc_a:.4f} (trained on {len(y_a)} samples)")
    print(f"{version_b}: {acc_b:.4f} (trained on {len(y_b)} samples)")
    print(f"Difference: {(acc_b - acc_a)*100:+.2f}%")

    diff = np.sum(pred_a != pred_b)
    if diff:
        print(f"\nModels disagree on {diff}/{len(pred_a)} test samples")
    return acc_a, acc_b

def version_key(v):
    """
    Sort versions intelligently:
    - version_0, version_1, version_2...
    - igt_P01, igt_P02, igt_P03...
    - *_trained versions come after their base
    """
    # Handle version_X format (version_0, version_1, etc.)
    match = re.search(r'version_(\d+)', v)
    if match:
        return (0, int(match.group(1)), '')  # version_ first
    
    # Handle igt_PXX format (igt_P01, igt_P02, etc.)
    match = re.search(r'igt_P(\d+)', v)
    if match:
        return (1, int(match.group(1)), '')  # igt_ second
    
    # Handle trained versions (igt_P01_trained, version_1_trained)
    match = re.search(r'(.+)_trained', v)
    if match:
        base = match.group(1)
        # Get base key recursively
        base_key = version_key(base)
        # Trained versions come after base (add 1 to third element)
        return (base_key[0], base_key[1], base_key[2] + 1)
    
    # Handle any other format (fallback)
    return (999, 0, v)

def check_current_version():
    """
    Check which versions exist and show the latest one
    """
    # Find all model files
    model_files = glob.glob('model_*.pkl')
    
    if not model_files:
        print("❌ No versions found")
        return None
    
    # Extract version names
    versions = []
    for f in model_files:
        # Extract version name (removes 'model_' and '.pkl')
        version = f.replace('model_', '').replace('.pkl', '')
        versions.append(version)
    
    # Sort versions using custom key
    versions.sort(key=version_key)
    
    print("\n📁 Available versions:")
    for v in versions:
        # Check if corresponding files exist
        has_data = os.path.exists(f'X_train_{v}.npy')
        has_model = os.path.exists(f'model_{v}.pkl')
        has_le = os.path.exists(f'label_encoder_{v}.pkl')
        
        status = "✅" if has_data and has_model and has_le else "⚠️"
        print(f"  {status} {v}")
    
    latest = versions[-1]
    print(f"\n👉 Latest version: {latest}")
    
    return versions

if __name__ == "__main__":
    # First show available versions
    versions = check_current_version()
    
    if versions and len(versions) >= 2:
        print("\n" + "="*60)
        print("📊 COMPARISON")
        print("="*60)
        # Compare oldest vs newest
        compare_versions(versions[0], versions[-1])
    else:
        print("\nNot enough versions to compare (need at least 2)")