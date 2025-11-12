import requests
import json

URL = "http://127.0.0.1:5000/predict"

test_data = {
    "lat": 24.39,
    "long": 84.76,
    "month": 10,
    "day": 27,
    "acq_hour": 14,
    "temp2m": 30,
    "humidity2m": 70,
    "precip": 0.1,
    "windspeed10m": 5,
    "cloudcover": 40
}

response = requests.post(URL, json=test_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
