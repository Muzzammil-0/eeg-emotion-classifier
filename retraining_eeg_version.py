import sys
import numpy as np
from model_utility_eeg import load_training_state, save_training_state
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def retrain_version(version_name, new_version_name=None):
    """
    Retrain a model on a saved version's data
    
    Parameters:
    - version_name: the version to retrain (e.g., 'version_1')
    - new_version_name: optional name for the trained version (default: version_name + '_trained')
    """
    print(f"\n Retraining version: {version_name}")
    
    # 1. Load the saved data (features + string labels)
    X, y, old_model, old_le, male_bl, female_bl = load_training_state(version_name)
    
    print(f"   Loaded {len(y)} training samples with {X.shape[1]} features")
    
    # 2. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 3. Recreate the exact same model architecture as original
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    rf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42)
    xgb = XGBClassifier(max_depth=3, n_estimators=50, random_state=42, n_jobs=-1)
    
    new_model = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 5]
    )
    
    # 4. Train on the data
    print("   Training model...")
    new_model.fit(X, y_enc)
    
    # 5. Save the retrained version
    out_version = new_version_name if new_version_name else f"{version_name}_trained"
    
    save_training_state(
        version_0=out_version,
        X=X,
        y=y,
        model=new_model,
        le=le,
        male_baseline=male_bl,
        female_baseline=female_bl
    )
    
    print(f" Retrained model saved as: {out_version}")
    
    # Optional: Quick evaluation on training data
    train_acc = new_model.score(X, y_enc)
    print(f"   Training accuracy: {train_acc:.4f}")
    
    return out_version

if __name__ == "__main__":
    if len(sys.argv) > 1:
        retrain_version(sys.argv[1])
    else:
        print("Usage: python retrain_model.py <version_name>")
        print("Example: python retrain_model.py version_1")

