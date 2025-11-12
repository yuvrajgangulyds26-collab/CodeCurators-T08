# test_fire_classifier.py
import joblib
import numpy as np
import math

# --- Load model and scaler ---
clf_data = joblib.load("fire_classification_model_2.pkl")
clf_model = clf_data["model"]
clf_scaler = clf_data["scaler"]
clf_features = clf_data["features"]

# --- Helper: Feature generator ---
def make_features(lat, lon, month, day, hour, temp, humidity, precip, wind, cloud):
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin = math.sin(2 * math.pi * day / 31)
    day_cos = math.cos(2 * math.pi * day / 31)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    X = np.array([[lat, lon,
                   month_sin, month_cos,
                   day_sin, day_cos,
                   hour_sin, hour_cos,
                   temp, humidity,
                   precip, wind,
                   cloud]])
    return X

# --- Test cases ---
test_cases = [
    # Tropical forest
    {"lat": 0, "lon": 30, "month": 10, "day": 15, "hour": 14, "temp": 30, "humidity": 80, "precip": 0, "wind": 5, "cloud": 40},
    # Snowy region
    {"lat": 70, "lon": 0, "month": 1, "day": 5, "hour": 12, "temp": -15, "humidity": 50, "precip": 0, "wind": 10, "cloud": 20},
    # Rainy tropical
    {"lat": -5, "lon": 120, "month": 11, "day": 20, "hour": 16, "temp": 28, "humidity": 95, "precip": 10, "wind": 2, "cloud": 90},
    # Desert
    {"lat": 25, "lon": 55, "month": 6, "day": 10, "hour": 11, "temp": 42, "humidity": 10, "precip": 0, "wind": 15, "cloud": 0}
]

# --- Run tests ---
for i, case in enumerate(test_cases):
    X = make_features(**case)
    X_scaled = clf_scaler.transform(X)
    pred = clf_model.predict(X_scaled)[0]
    prob = clf_model.predict_proba(X_scaled)[0]
    label = "Fire" if pred == 1 else "No Fire"
    print(f"Test Case {i+1}: {case}")
    print(f"  Predicted Class: {label}, Probabilities [No Fire, Fire]: {prob}\n")
