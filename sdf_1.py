import requests

BASE_URL = "https://universal-eeg-emotion-classifier.onrender.com"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test prediction
files = {'file': open(r'C:\Users\hp\Downloads\ds007338\ds007338\sub-EP10\ses-01\eeg\sub-EP10_ses-01_task-dots_run-01_eeg.edf', 'rb')}
response = requests.post(f"{BASE_URL}/predict", files=files)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")