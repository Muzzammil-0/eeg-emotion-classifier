import requests

# Use localhost for local testing
BASE_URL = "http://127.0.0.1:5000"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test prediction with your local Flask
files = {'file': open({'https://universal-eeg-emotion-classifier.onrender.com/'}, 'rb')}
response = requests.post(f"{BASE_URL}/upload", files=files)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")
