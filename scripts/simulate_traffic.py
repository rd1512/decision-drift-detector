import time
import requests
import random

URL = "http://127.0.0.1:8000/predict"

def generate_features():
    return {
        "features": [random.random() for _ in range(30)]
    }

while True:
    response = requests.post(URL, json=generate_features())
    print(response.json())
    time.sleep(2)  # one prediction every 2 seconds
