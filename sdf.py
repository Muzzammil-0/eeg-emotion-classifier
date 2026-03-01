
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'EEG Emotion Classifier API is running in minimal test mode.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f" Minimal test server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)