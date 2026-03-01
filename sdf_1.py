import requests

# Test health
response = requests.get("http://127.0.0.1:5000/health")
print("Health:", response.json())

# Test prediction
files = {'file': open(r'C:\Users\hp\Downloads\ds007338\ds007338\sub-EP10\ses-01\eeg\sub-EP10_ses-01_task-dots_run-01_eeg.edf', 'rb')}
response = requests.post("http://127.0.0.1:5000/predict", files=files)
print("Prediction:", response.json())
