import requests
import json

# URL of your locally running Flask server
url = "http://localhost:5001/predict"

# Sample input data (must match the model's expected input format)
payload = [
    {
        "cylinders": 4,
        "displacement": 140.0,
        "horsepower": 90.0,
        "weight": 2264.0,
        "acceleration": 15.5,
        "model_year": 78,
        "origin_2": 0,
        "origin_3": 1
    },
    {
        "cylinders": 6,
        "displacement": 250.0,
        "horsepower": 105.0,
        "weight": 3353.0,
        "acceleration": 14.5,
        "model_year": 77,
        "origin_2": 1,
        "origin_3": 0
    }
]

# Send POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Print response
if response.ok:
    print("Predictions:", response.json())
else:
    print("Error:", response.status_code, response.text)
