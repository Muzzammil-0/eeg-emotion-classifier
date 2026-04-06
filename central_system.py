
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# File to store aggregated features (CSV)
DATA_CSV = 'global_features.csv'
MODEL_FILE = 'global_model.pkl'
LE_FILE = 'global_label_encoder.pkl'

# Ensure CSV exists with headers (we'll create on first upload)
def init_csv():
    if not os.path.exists(DATA_CSV):
        # We need to know feature names. We'll get them from first upload.
        # For now, create empty.
        open(DATA_CSV, 'w').close()

init_csv()

@app.route('/upload_features', methods=['POST'])
def upload_features():
    """Receive a list of feature vectors + labels from a client."""
    data = request.get_json()
    # Expected format: {"features": [list of floats], "label": "POSITIVE", "client_id": "optional"}
    features = data['features']
    label = data['label']
    
    # Convert to DataFrame row
    # We don't know column names yet; we'll store as a string or generic columns.
    # Better: store as a row in CSV with generic column names f0..f17 + label.
    # First, determine number of features (should be 18)
    n_features = len(features)
    col_names = [f'f{i}' for i in range(n_features)] + ['label']
    
    df_new = pd.DataFrame([features + [label]], columns=col_names)
    # Append to CSV, create header if file is empty
    df_new.to_csv(DATA_CSV, mode='a', header=not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV)==0, index=False)
    
    return jsonify({"status": "ok", "received": len(features)})

@app.route('/retrain_global', methods=['POST'])
def retrain_global():
    """Retrain the global ensemble on all collected features."""
    if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
        return jsonify({"error": "No data available"}), 400
    
    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, :-1].values  # all feature columns
    y = df['label'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Build same ensemble as in dipps.py
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    rf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42)
    xgb = XGBClassifier(max_depth=3, n_estimators=50, random_state=42, n_jobs=-1)
    model = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 5]
    )
    model.fit(X, y_enc)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, LE_FILE)
    
    return jsonify({"message": f"Global model retrained on {len(df)} samples", "samples": len(df)})

@app.route('/download_model', methods=['GET'])
def download_model():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "No global model yet"}), 404
    return send_file(MODEL_FILE, as_attachment=True)

@app.route('/download_label_encoder', methods=['GET'])
def download_le():
    if not os.path.exists(LE_FILE):
        return jsonify({"error": "No label encoder yet"}), 404
    return send_file(LE_FILE, as_attachment=True)

@app.route('/stats', methods=['GET'])
def stats():
    if not os.path.exists(DATA_CSV):
        return jsonify({"samples": 0})
    df = pd.read_csv(DATA_CSV)
    return jsonify({"samples": len(df)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

