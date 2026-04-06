
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
from functools import wraps

app = Flask(__name__)
CORS(app)

# ---------- Authentication ----------
API_KEY = "your-secret-key-123"   # MUST match the key used in sdf.py

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ---------- File paths ----------
DATA_CSV = 'global_features.csv'
MODEL_FILE = 'global_model.pkl'
LE_FILE = 'global_label_encoder.pkl'

# ---------- Helper ----------
def get_sample_count():
    if not os.path.exists(DATA_CSV):
        return 0
    try:
        df = pd.read_csv(DATA_CSV)
        return len(df)
    except:
        return 0

# ---------- Routes ----------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "alive", "samples": get_sample_count()})

@app.route('/upload_batch', methods=['POST'])
@require_api_key
def upload_batch():
    """Receive a batch of feature vectors + labels."""
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list of samples"}), 400
    if not data:
        return jsonify({"status": "ok", "uploaded": 0})

    n_features = len(data[0]['features'])
    rows = []
    for item in data:
        rows.append(item['features'] + [item['label']])

    df_new = pd.DataFrame(rows, columns=[f'f{i}' for i in range(n_features)] + ['label'])
    file_exists = os.path.exists(DATA_CSV) and os.path.getsize(DATA_CSV) > 0
    df_new.to_csv(DATA_CSV, mode='a', header=not file_exists, index=False)

    return jsonify({"status": "ok", "uploaded": len(data)})

@app.route('/retrain_global', methods=['POST'])
@require_api_key
def retrain_global():
    if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
        return jsonify({"error": "No data available"}), 400

    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, :-1].values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

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
@require_api_key
def download_model():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "No global model yet"}), 404
    return send_file(MODEL_FILE, as_attachment=True)

@app.route('/download_label_encoder', methods=['GET'])
@require_api_key
def download_le():
    if not os.path.exists(LE_FILE):
        return jsonify({"error": "No label encoder yet"}), 404
    return send_file(LE_FILE, as_attachment=True)

@app.route('/stats', methods=['GET'])
@require_api_key
def stats():
    return jsonify({"samples": get_sample_count()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
