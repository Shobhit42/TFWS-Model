import requests
import json

# Define the server URL
server_url = 'http://localhost:5000/predict'

# Sample input data
input_data = {"features": [23, 11500, 13.33, 2, 0, 5.0]}

# Send a POST request
response = requests.post(server_url, json=input_data)

# Print the response
if response.status_code == 200:
    data = response.json()
    print("Prediction:", data['prediction'])
    print("Probability:", data['probability'])
else:
    print("Error:", response.status_code, response.text)
