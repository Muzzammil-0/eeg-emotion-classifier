import requests

# Use localhost for local testing
BASE_URL = "http://127.0.0.1:10000"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test prediction with your local Flask
files = {'file': open(r'C:\Users\hp\Downloads\ds007338\ds007338\sub-EP10\ses-01\eeg\sub-EP10_ses-01_task-dots_run-01_eeg.edf', 'rb')}
response = requests.post(f"{BASE_URL}/predict", files=files)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")
