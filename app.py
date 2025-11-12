from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS
import joblib
import numpy as np
import math
import requests
from datetime import datetime
import sqlite3
import re
import random
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ===========================================================
#  App Setup
# ===========================================================
app = Flask(__name__)
CORS(app)
app.secret_key = "super_secret_key_123"  # change to environment variable in production

# ===========================================================
#  Database Setup (users.db)
# ===========================================================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # users table (already present)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    # predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            fire_class INTEGER NOT NULL,
            predicted_frp REAL NOT NULL,
            weather_json TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ===========================================================
#  OTP Generation Route
# ===========================================================
@app.route("/generate_otp", methods=["POST"])
def generate_otp():
    try:
        data = request.get_json()
        email = data.get("email")
        if not email:
            return jsonify({"error": "Email required"}), 400

        otp = random.randint(100000, 999999)
        session["otp"] = otp
        session["otp_email"] = email

        # In real production use an email service — here we just log it
        print(f"Generated OTP for {email}: {otp}")
        return jsonify({"otp": otp})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===========================================================
#  Auth Helpers
# ===========================================================
def valid_email(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)

def strong_password(password):
    return (
        len(password) >= 8
        and re.search(r"[A-Z]", password)
        and re.search(r"[a-z]", password)
        and re.search(r"\d", password)
        and re.search(r"[@$!%*?&]", password)
    )

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the dashboard.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ===========================================================
#  Model Loading
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
#  Weather & Feature Builders (same as before)
# ===========================================================
def get_weather(lat, lon, timestamp=None):
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

        if "hourly" not in data:
            raise KeyError("No hourly data")

        idx = -1
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
        return {
            "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0,
            "precipitation": 0.0,
            "cloudcover": 40.0,
            "windspeed_10m": 5.0,
            "source": "fallback"
        }

def get_satellite_defaults():
    return {
        "brightness": 3.288921e+02,
        "bright_t31": 3.001991e+02,
        "confidence": 6.833371e+01,
        "distance_km": None,
        "source": "mean_defaults"
    }

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
    if len(features) < 44:
        features.extend([0] * (44 - len(features)))
    return features

def build_regression_features(lat, lon, weather, sat, ts):
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    is_northern = 1 if lat >= 0 else 0
    is_eastern = 1 if lon >= 0 else 0

    zone_equatorial = 1 if abs(lat) < 15 else 0
    zone_temperate = 1 if 15 <= abs(lat) < 45 else 0
    zone_polar = 1 if abs(lat) >= 45 else 0

    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour

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

    return [
        lat, lon, lat_abs, lon_abs, is_northern, is_eastern,
        zone_equatorial, zone_temperate, zone_polar,
        month_sin, month_cos, day_sin, day_cos,
        hour_sin, hour_cos, temp, humidity,
        precip, wind, cloud, dryness_index, wind_precip_index,
        temp_cloud_diff, brightness, bright_t31, confidence
    ]

# ===========================================================
#  Auth Routes
# ===========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form["email"]
        password = request.form["password"]
        otp_input = request.form.get("otp")

        print("FORM DATA:", request.form)
        print("SESSION OTP:", session.get("otp"))

        # Check OTP match
        if otp_input != str(session.get("otp")) or email != session.get("otp_email"):
            flash("Invalid OTP or email mismatch!", "error")
            return redirect(url_for("signup"))

        if not valid_email(email):
            flash("Invalid email format.", "warning")
            return redirect(url_for("signup"))

        if not strong_password(password):
            flash("Password too weak (need upper, lower, digit, symbol, min 8 chars).", "warning")
            return redirect(url_for("signup"))

        password_hash = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (fullname, email, password_hash) VALUES (?, ?, ?)",
                           (fullname, email, password_hash))
            conn.commit()

            # Fetch user ID to store in session for auto-login
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

            session["user_id"] = user[0]
            session["email"] = email
            flash("Signup successful! Welcome to your dashboard.", "success")
            return redirect(url_for("dashboard"))

        except sqlite3.IntegrityError:
            flash("Email already registered. Please log in instead.", "danger")
            return redirect(url_for("login"))
        finally:
            conn.close()

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["email"] = email
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "info")
    return redirect(url_for("index"))

# ===========================================================
#  Protected Dashboard
# ===========================================================
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# ===========================================================
#  History endpoints
# ===========================================================
@app.route("/history")
@login_required
def history():
    # renders the history page which will fetch data from /get_history via JS
    return render_template("history.html")

@app.route("/get_history")
@login_required
def get_history():
    user_id = session.get("user_id")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, latitude, longitude, fire_class, predicted_frp, weather_json, timestamp
            FROM predictions
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        rows = cursor.fetchall()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "id": r[0],
                "latitude": r[1],
                "longitude": r[2],
                "fire_class": int(r[3]),
                "predicted_frp": float(r[4]),
                "weather": r[5],
                "timestamp": r[6]
            })
        return jsonify(data)
    except Exception as e:
        print("History fetch error:", e)
        return jsonify([])

# ===========================================================
#  Predict route (stores prediction in DB)
# ===========================================================
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.get_json()
        lat = float(data["lat"])
        lon = float(data["lon"])
        timestamp = data.get("timestamp")

        weather = get_weather(lat, lon, timestamp)
        sat = get_satellite_defaults()

        X_ens = build_ensemble_features(lat, lon, weather, timestamp)
        X_reg = build_regression_features(lat, lon, weather, sat, timestamp)

        fire_class = voting_model.predict([X_ens])[0]
        fire_label = "Fire" if int(fire_class) == 1 else "No Fire"

        X_reg_scaled = reg_scaler.transform([X_reg])
        frp_pred = reg_model.predict(X_reg_scaled)[0]

        # --- store prediction in DB ---
        try:
            user_id = session.get("user_id")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (user_id, latitude, longitude, fire_class, predicted_frp, weather_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                lat,
                lon,
                int(fire_class),
                float(frp_pred),
                str(weather),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as db_e:
            # log DB write failures but do not prevent returning prediction
            print("Prediction DB write error:", db_e)

        return jsonify({
            "fire_label": fire_label,
            "fire_class": int(fire_class),
            "predicted_frp": round(float(frp_pred), 2),
            "weather": weather
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400

# ===========================================================
#  Run Server
# ===========================================================
if __name__ == "__main__":
    app.run(debug=True)
