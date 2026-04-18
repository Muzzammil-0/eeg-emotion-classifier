import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import json
import os
import sys

sys.path.append(r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER')
from dipps import bins_to_waves

DATA_PATH  = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER\emotions_27.csv'
OUTPUT_DIR = r'C:\Users\PROJECTS\EEG_EMOTION_CLASSIFIER'


def build_model():
    sklearn.set_config(enable_metadata_routing=True)
    dt = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
         ).set_fit_request(sample_weight=True)
    rf = RandomForestClassifier(
            max_depth=3,
            n_estimators=50,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
         ).set_fit_request(sample_weight=True)
    xgb = XGBClassifier(
            max_depth=3,
            n_estimators=50,
            min_child_weight=10,
            random_state=42,
            n_jobs=-1
          ).set_fit_request(sample_weight=True)
    return VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 5]
    )


def run_loso(data_path, output_dir):
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    fft_cols = ([f'fft_{i}_a' for i in range(500)] +
                [f'fft_{i}_b' for i in range(500)])

    subjects = sorted(df['subject_id'].unique())

    le = LabelEncoder()
    le.fit(df['label'])

    all_results = []
    all_true    = []
    all_pred    = []

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(subjects)} — Test: {test_subject}")
        print(f"{'='*60}")

        train_df = df[df['subject_id'] != test_subject].copy()
        test_df  = df[df['subject_id'] == test_subject].copy()

        print(f"Train: {len(train_df)} | Test: {len(test_df)}")

        # Impute
        imputer = SimpleImputer(strategy='mean')
        train_df[fft_cols] = imputer.fit_transform(train_df[fft_cols])
        test_df[fft_cols]  = imputer.transform(test_df[fft_cols])

        # Scale
        scaler = MinMaxScaler()
        train_df[fft_cols] = scaler.fit_transform(train_df[fft_cols])
        test_df[fft_cols]  = scaler.transform(test_df[fft_cols])

        # Band features
        x_train = bins_to_waves(train_df[fft_cols])
        x_test  = bins_to_waves(test_df[fft_cols])

        y_train = le.transform(train_df['label'])
        y_test  = le.transform(test_df['label'])

        # Sample weights
        sample_weights = compute_sample_weight('balanced', y=y_train)

        # Train
        model = build_model()
        model.fit(x_train, y_train, sample_weight=sample_weights)

        # Evaluate
        y_pred    = model.predict(x_test)
        bal_acc   = balanced_accuracy_score(y_test, y_pred)
        train_acc = model.score(x_train, y_train)

        print(f"Train acc:    {train_acc:.4f}")
        print(f"Balanced acc: {bal_acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        all_results.append({
            'subject':      test_subject,
            'train_acc':    round(train_acc, 4),
            'balanced_acc': round(bal_acc, 4),
            'n_test':       len(test_df)
        })
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    # Overall
    print(f"\n{'='*60}")
    print("LOSO CROSS VALIDATION — OVERALL RESULTS")
    print(f"{'='*60}")

    mean_bal = np.mean([r['balanced_acc'] for r in all_results])
    std_bal  = np.std([r['balanced_acc']  for r in all_results])

    print(f"Mean balanced accuracy: {mean_bal:.4f} ± {std_bal:.4f}")
    print(f"\nOverall classification report:")
    print(classification_report(all_true, all_pred, target_names=le.classes_))

    results_df = pd.DataFrame(all_results).sort_values('balanced_acc', ascending=False)
    print(f"\nPer-subject results:")
    print(results_df.to_string(index=False))

    # Save
    results_path = os.path.join(output_dir, 'loso_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'mean_balanced_accuracy': round(mean_bal, 4),
            'std_balanced_accuracy':  round(std_bal, 4),
            'per_subject':            all_results,
            'overall_report':         classification_report(
                                          all_true, all_pred,
                                          target_names=le.classes_,
                                          output_dict=True)
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")
    return mean_bal, std_bal


if __name__ == '__main__':
    mean_acc, std_acc = run_loso(DATA_PATH, OUTPUT_DIR)
    print(f"\nFinal LOSO score: {mean_acc:.4f} ± {std_acc:.4f}")