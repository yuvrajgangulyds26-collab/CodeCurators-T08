from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import math
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===========================================================
#  Load Models
# ===========================================================
print("Loading models...")

voting_model = joblib.load("voting_ensemble_fire.joblib")
print("Voting Ensemble loaded successfully!")

reg_data = joblib.load("fire_regression_model_2.pkl")
reg_model = reg_data["model"]
reg_scaler = reg_data["scaler"]
reg_features = reg_data["features"]
print("Regression model loaded successfully!")


# ===========================================================
#  Helper Functions
# ===========================================================
def get_weather(lat, lon, timestamp=None):
    """Fetch current or historical weather data from Open-Meteo API."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloudcover,windspeed_10m",
            "timezone": "auto",
        }
        url = "https://api.open-meteo.com/v1/forecast"
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        # Safely handle missing "hourly"
        if "hourly" not in data:
            print("Weather API missing 'hourly', using fallback.")
            raise KeyError("No hourly data")

        idx = -1  # latest available hour
        weather = {
            "temperature_2m": float(data["hourly"]["temperature_2m"][idx]),
            "relative_humidity_2m": float(data["hourly"]["relative_humidity_2m"][idx]),
            "precipitation": float(data["hourly"]["precipitation"][idx]),
            "cloudcover": float(data["hourly"]["cloudcover"][idx]),
            "windspeed_10m": float(data["hourly"]["windspeed_10m"][idx]),
            "source": "open-meteo"
        }
        return weather

    except Exception as e:
        print("Weather fetch error:", e)
        # Fallback defaults
        return {
            "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0,
            "precipitation": 0.0,
            "cloudcover": 40.0,
            "windspeed_10m": 5.0,
            "source": "fallback"
        }


def get_satellite_defaults():
    """Return constant mean FIRMS-like satellite values (since unused in training)."""
    return {
        "brightness": 3.288921e+02,
        "bright_t31": 3.001991e+02,
        "confidence": 6.833371e+01,
        "distance_km": None,
        "source": "mean_defaults"
    }


# ===========================================================
#  Feature Builders
# ===========================================================
def build_ensemble_features(lat, lon, weather, ts):
    now = datetime.now()
    month = now.month
    day = now.day
    year = now.year

    FFMC = DMC = DC = ISI = area = 0
    temp = weather["temperature_2m"]
    RH = weather["relative_humidity_2m"]
    wind = weather["windspeed_10m"]
    rain = weather["precipitation"]

    # Derived features
    temp_RH = temp * RH
    wind_ISI = wind * ISI
    rain_DMC = rain * DMC
    FFMC_ISI = FFMC * ISI
    temp_RH_ratio = temp / (RH + 1)
    wind_rain_ratio = wind / (rain + 1)
    FFMC_sq = FFMC ** 2
    temp_sq = temp ** 2
    wind_sq = wind ** 2
    ISI_log = math.log(ISI + 1)
    area_log = math.log(area + 1)

    features = [
        lon, lat, month, day, FFMC, DMC, DC, ISI,
        temp, RH, wind, rain, area,
        month, day, year, month * 31 + day,
        temp_RH, wind_ISI, rain_DMC, FFMC_ISI,
        temp_RH_ratio, wind_rain_ratio,
        FFMC_sq, temp_sq, wind_sq, ISI_log, area_log
    ]

    # Pad to 44 features
    if len(features) < 44:
        features.extend([0] * (44 - len(features)))

    return features


def build_regression_features(lat, lon, weather, sat, ts):
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    is_northern = 1 if lat >= 0 else 0
    is_eastern = 1 if lon >= 0 else 0

    # Climate zones
    zone_equatorial = 1 if abs(lat) < 15 else 0
    zone_temperate = 1 if 15 <= abs(lat) < 45 else 0
    zone_polar = 1 if abs(lat) >= 45 else 0

    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour

    # Cyclic encodings
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin = math.sin(2 * math.pi * day / 31)
    day_cos = math.cos(2 * math.pi * day / 31)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    temp = weather["temperature_2m"]
    humidity = weather["relative_humidity_2m"]
    precip = weather["precipitation"]
    wind = weather["windspeed_10m"]
    cloud = weather["cloudcover"]

    dryness_index = temp - humidity / 5
    wind_precip_index = wind / (1 + precip)
    temp_cloud_diff = temp - cloud / 10

    brightness = sat["brightness"]
    bright_t31 = sat["bright_t31"]
    confidence = sat["confidence"]

    reg_vector = [
        lat, lon, lat_abs, lon_abs, is_northern, is_eastern,
        zone_equatorial, zone_temperate, zone_polar,
        month_sin, month_cos, day_sin, day_cos,
        hour_sin, hour_cos, temp, humidity,
        precip, wind, cloud, dryness_index, wind_precip_index,
        temp_cloud_diff, brightness, bright_t31, confidence
    ]

    return reg_vector


# ===========================================================
#  Prediction Route
# ===========================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        lat = float(data["lat"])
        lon = float(data["lon"])
        timestamp = data.get("timestamp")

        print("Received payload:", data)

        # Step 1: Weather + Satellite (mean)
        weather = get_weather(lat, lon, timestamp)
        sat = get_satellite_defaults()

        # Step 2: Build features
        X_ens = build_ensemble_features(lat, lon, weather, timestamp)
        X_reg = build_regression_features(lat, lon, weather, sat, timestamp)

        # Step 3: Predictions
        fire_class = voting_model.predict([X_ens])[0]
        fire_label = "Fire" if int(fire_class) == 1 else "No Fire"

        X_reg_scaled = reg_scaler.transform([X_reg])
        frp_pred = reg_model.predict(X_reg_scaled)[0]

        # Step 4: Response
        response = {
            "fire_label": fire_label,
            "fire_class": int(fire_class),
            "predicted_frp": round(float(frp_pred), 2),
            "weather": weather
        }

        print("Response:", response)
        return jsonify(response)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400


# ===========================================================
#  Run Server
# ===========================================================
if __name__ == "__main__":
    app.run(debug=True)
