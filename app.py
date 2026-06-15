import os
import json
import math
import random
import re
import smtplib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

import joblib
import numpy as np
import requests
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# ===========================================================
#  Load Environment Variables
# ===========================================================
load_dotenv()

SECRET_KEY    = os.environ.get("SECRET_KEY", "fallback-dev-key-change-in-production")
GMAIL_USER    = os.environ.get("GMAIL_USER", "wildfirepredictor@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD")   # set in .env — never hardcode
DB_PATH       = os.environ.get("DB_PATH", "users.db")
DEMO_MODE     = os.environ.get("DEMO_MODE", "false").lower() == "true"
DEMO_OTP      = 123456   # fixed OTP shown to user in demo mode

# ===========================================================
#  Model Download — fetch from HuggingFace Hub if not local
# ===========================================================
_HF_REPO = "yuvrajganguly/wildfire-models"
_MODEL_FILES = [
    "voting_ensemble_fire.joblib",
    "fire_regression_model_2.pkl",
]

def _ensure_models() -> None:
    """Download model files from HuggingFace Hub when running in production."""
    try:
        from huggingface_hub import hf_hub_download
        for filename in _MODEL_FILES:
            if not os.path.exists(filename):
                print(f"[startup] Downloading {filename} from HuggingFace Hub ...")
                hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=filename,
                    local_dir=".",
                )
                print(f"[startup] Downloaded {filename}")
            else:
                print(f"[startup] Model already present: {filename}")
    except Exception as e:
        print(f"[startup] HuggingFace model download failed: {e} (models must be present locally)")

_ensure_models()

# ===========================================================
#  Logging Setup
# ===========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wildfire")

# ===========================================================
#  App Setup
# ===========================================================
app = Flask(__name__)
CORS(app)
app.secret_key = SECRET_KEY

# Rate limiter — limits abusive calls to sensitive endpoints
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],          # no global limit; applied per route
    storage_uri="memory://",
)

# ===========================================================
#  Database — Context Manager Helper
# ===========================================================
@contextmanager
def get_db():
    """Yields a (conn, cursor) pair and guarantees cleanup."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    try:
        cursor = conn.cursor()
        yield conn, cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as (conn, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname      TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                latitude      REAL    NOT NULL,
                longitude     REAL    NOT NULL,
                fire_class    INTEGER NOT NULL,
                predicted_frp REAL    NOT NULL,
                weather_json  TEXT,
                timestamp     TEXT    NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        # Index for fast per-user history lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_user_id
            ON predictions(user_id)
        """)

init_db()
logger.info("Database initialised.")

# ===========================================================
#  Email — OTP Sender
# ===========================================================
OTP_EXPIRY_SECONDS = 300   # 5 minutes

def send_otp_email(to_email: str, otp: int) -> bool:
    """Send a 6-digit OTP to to_email via Gmail SMTP. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your Wildfire Predictor OTP"
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_email

        body = f"""\
Hi,

Your one-time verification code is:

    {otp}

This code expires in 5 minutes. Do not share it with anyone.

— Wildfire Predictor Team
"""
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        logger.info("OTP email sent to %s", to_email)
        return True

    except Exception as e:
        logger.error("Failed to send OTP email to %s: %s", to_email, e)
        return False

# ===========================================================
#  OTP Route — rate limited, OTP never returned to client
# ===========================================================
@app.route("/generate_otp", methods=["POST"])
@limiter.limit("5 per minute")
def generate_otp():
    try:
        data  = request.get_json()
        email = (data or {}).get("email", "").strip()

        if not email:
            return jsonify({"error": "Email required"}), 400

        if not valid_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        otp = DEMO_OTP if DEMO_MODE else random.randint(100000, 999999)

        # Store OTP + generation time in session; never send OTP to client
        session["otp"]              = otp
        session["otp_email"]        = email
        session["otp_generated_at"] = datetime.now(timezone.utc).isoformat()
        session["otp_used"]         = False

        if DEMO_MODE:
            return jsonify({"message": f"Demo mode: use OTP {DEMO_OTP} (no email sent)"})

        success = send_otp_email(email, otp)
        if not success:
            return jsonify({"error": "Failed to send OTP. Check email address."}), 500

        return jsonify({"message": "OTP sent to your email."})

    except Exception as e:
        logger.error("generate_otp error: %s", e)
        return jsonify({"error": "Internal server error"}), 500

# ===========================================================
#  Auth Helpers
# ===========================================================
def valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email))

def strong_password(password: str) -> bool:
    return (
        len(password) >= 8
        and bool(re.search(r"[A-Z]", password))
        and bool(re.search(r"[a-z]", password))
        and bool(re.search(r"\d",    password))
        and bool(re.search(r"[@$!%*?&]", password))
    )

def verify_otp(email: str, otp_input: str) -> tuple[bool, str]:
    """
    Verifies OTP from session.
    Returns (True, "") on success or (False, reason) on failure.
    """
    stored_otp   = session.get("otp")
    stored_email = session.get("otp_email")
    generated_at = session.get("otp_generated_at")
    used         = session.get("otp_used", True)

    if used:
        return False, "OTP has already been used."

    if stored_otp is None or stored_email is None or generated_at is None:
        return False, "No OTP found. Please request a new one."

    if email != stored_email:
        return False, "Email does not match OTP recipient."

    # Check expiry
    generated_dt = datetime.fromisoformat(generated_at)
    elapsed      = (datetime.now(timezone.utc) - generated_dt).total_seconds()
    if elapsed > OTP_EXPIRY_SECONDS:
        return False, "OTP has expired. Please request a new one."

    if otp_input != str(stored_otp):
        return False, "Incorrect OTP."

    # Mark as used immediately
    session["otp_used"] = True
    return True, ""

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ===========================================================
#  Model Loading
# ===========================================================
logger.info("Loading ML models...")

try:
    voting_model = joblib.load("voting_ensemble_fire.joblib")
    logger.info("Voting ensemble model loaded.")
except FileNotFoundError:
    logger.critical("voting_ensemble_fire.joblib not found — server cannot start.")
    raise

try:
    reg_data     = joblib.load("fire_regression_model_2.pkl")
    reg_model    = reg_data["model"]
    reg_scaler   = reg_data["scaler"]
    reg_features = reg_data["features"]
    logger.info("Regression model loaded.")
except FileNotFoundError:
    logger.critical("fire_regression_model_2.pkl not found — server cannot start.")
    raise
except KeyError as e:
    logger.critical("Regression model .pkl missing key: %s", e)
    raise

# ===========================================================
#  Weather Helper — picks closest hour to now
# ===========================================================
def get_weather(lat: float, lon: float) -> dict:
    """Fetch weather from Open-Meteo for the hour closest to now."""
    try:
        params = {
            "latitude":  lat,
            "longitude": lon,
            "hourly":    "temperature_2m,relative_humidity_2m,precipitation,cloudcover,windspeed_10m",
            "timezone":  "auto",
        }
        res  = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if "hourly" not in data:
            raise KeyError("No hourly data in response")

        times = data["hourly"]["time"]           # list of "YYYY-MM-DDTHH:00" strings
        now   = datetime.now()

        # Find the index of the closest hour
        def hour_diff(t_str):
            t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
            return abs((t - now).total_seconds())

        idx = min(range(len(times)), key=lambda i: hour_diff(times[i]))

        weather = {
            "temperature_2m":       float(data["hourly"]["temperature_2m"][idx]),
            "relative_humidity_2m": float(data["hourly"]["relative_humidity_2m"][idx]),
            "precipitation":        float(data["hourly"]["precipitation"][idx]),
            "cloudcover":           float(data["hourly"]["cloudcover"][idx]),
            "windspeed_10m":        float(data["hourly"]["windspeed_10m"][idx]),
            "source":               "open-meteo",
        }
        logger.info("Weather fetched for (%.4f, %.4f) at hour index %d", lat, lon, idx)
        return weather

    except Exception as e:
        logger.warning("Weather fetch failed for (%.4f, %.4f): %s — using fallback", lat, lon, e)
        return {
            "temperature_2m":       25.0,
            "relative_humidity_2m": 50.0,
            "precipitation":        0.0,
            "cloudcover":           40.0,
            "windspeed_10m":        5.0,
            "source":               "fallback",
        }

# ===========================================================
#  Satellite Defaults
#  (placeholder — replace with NASA FIRMS API in future)
# ===========================================================
def get_satellite_defaults() -> dict:
    return {
        "brightness":  3.288921e+02,
        "bright_t31":  3.001991e+02,
        "confidence":  6.833371e+01,
        "source":      "mean_defaults",
    }

# ===========================================================
#  Feature Builders
# ===========================================================
ENSEMBLE_FEATURE_COUNT = 44

def build_ensemble_features(lat: float, lon: float, weather: dict) -> list:
    now   = datetime.now()
    month = now.month
    day   = now.day
    year  = now.year

    # Forest-fire index placeholders (no live source yet)
    FFMC = DMC = DC = ISI = area = 0

    temp = weather["temperature_2m"]
    RH   = weather["relative_humidity_2m"]
    wind = weather["windspeed_10m"]
    rain = weather["precipitation"]

    # Interaction & derived features
    temp_RH          = temp * RH
    wind_ISI         = wind * ISI
    rain_DMC         = rain * DMC
    FFMC_ISI         = FFMC * ISI
    temp_RH_ratio    = temp / (RH + 1)
    wind_rain_ratio  = wind / (rain + 1)
    FFMC_sq          = FFMC ** 2
    temp_sq          = temp ** 2
    wind_sq          = wind ** 2
    ISI_log          = math.log(ISI + 1)
    area_log         = math.log(area + 1)

    features = [
        lon, lat, month, day, FFMC, DMC, DC, ISI,
        temp, RH, wind, rain, area,
        month, day, year, month * 31 + day,
        temp_RH, wind_ISI, rain_DMC, FFMC_ISI,
        temp_RH_ratio, wind_rain_ratio,
        FFMC_sq, temp_sq, wind_sq, ISI_log, area_log,
    ]

    # Pad to expected count
    if len(features) < ENSEMBLE_FEATURE_COUNT:
        features.extend([0] * (ENSEMBLE_FEATURE_COUNT - len(features)))

    assert len(features) == ENSEMBLE_FEATURE_COUNT, (
        f"Ensemble feature count mismatch: expected {ENSEMBLE_FEATURE_COUNT}, got {len(features)}"
    )
    return features


REGRESSION_FEATURE_COUNT = 26

def build_regression_features(lat: float, lon: float, weather: dict, sat: dict) -> list:
    lat_abs      = abs(lat)
    lon_abs      = abs(lon)
    is_northern  = 1 if lat >= 0 else 0
    is_eastern   = 1 if lon >= 0 else 0

    zone_equatorial = 1 if abs(lat) < 15 else 0
    zone_temperate  = 1 if 15 <= abs(lat) < 45 else 0
    zone_polar      = 1 if abs(lat) >= 45 else 0

    now   = datetime.now()
    month = now.month
    day   = now.day
    hour  = now.hour

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin   = math.sin(2 * math.pi * day   / 31)
    day_cos   = math.cos(2 * math.pi * day   / 31)
    hour_sin  = math.sin(2 * math.pi * hour  / 24)
    hour_cos  = math.cos(2 * math.pi * hour  / 24)

    temp     = weather["temperature_2m"]
    humidity = weather["relative_humidity_2m"]
    precip   = weather["precipitation"]
    wind     = weather["windspeed_10m"]
    cloud    = weather["cloudcover"]

    dryness_index    = temp - humidity / 5
    wind_precip_idx  = wind / (1 + precip)
    temp_cloud_diff  = temp - cloud / 10

    brightness = sat["brightness"]
    bright_t31 = sat["bright_t31"]
    confidence = sat["confidence"]

    features = [
        lat, lon, lat_abs, lon_abs, is_northern, is_eastern,
        zone_equatorial, zone_temperate, zone_polar,
        month_sin, month_cos, day_sin, day_cos,
        hour_sin, hour_cos, temp, humidity,
        precip, wind, cloud, dryness_index, wind_precip_idx,
        temp_cloud_diff, brightness, bright_t31, confidence,
    ]

    assert len(features) == REGRESSION_FEATURE_COUNT, (
        f"Regression feature count mismatch: expected {REGRESSION_FEATURE_COUNT}, got {len(features)}"
    )
    return features

# ===========================================================
#  Forecast Weather Helper — returns next 24 hourly slots
# ===========================================================
def get_forecast_weather(lat: float, lon: float) -> list[dict]:
    """
    Fetch the next 24 hours of weather from Open-Meteo.
    Returns a list of dicts, one per hour, each with the same
    keys as get_weather() plus 'hour_dt' (a datetime object).
    Falls back to 24 copies of default weather on failure.
    """
    try:
        params = {
            "latitude":  lat,
            "longitude": lon,
            "hourly":    "temperature_2m,relative_humidity_2m,precipitation,cloudcover,windspeed_10m",
            "timezone":  "auto",
            "forecast_days": 2,   # gives ~48 hours so we always have 24 ahead
        }
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()

        if "hourly" not in data:
            raise KeyError("No hourly data in response")

        times = data["hourly"]["time"]
        now   = datetime.now()

        # Find the index of the first hour >= now
        def to_dt(t_str):
            return datetime.strptime(t_str, "%Y-%m-%dT%H:%M")

        start_idx = next(
            (i for i, t in enumerate(times) if to_dt(t) >= now),
            0
        )

        slots = []
        for i in range(start_idx, min(start_idx + 24, len(times))):
            slots.append({
                "hour_dt":              to_dt(times[i]),
                "temperature_2m":       float(data["hourly"]["temperature_2m"][i]),
                "relative_humidity_2m": float(data["hourly"]["relative_humidity_2m"][i]),
                "precipitation":        float(data["hourly"]["precipitation"][i]),
                "cloudcover":           float(data["hourly"]["cloudcover"][i]),
                "windspeed_10m":        float(data["hourly"]["windspeed_10m"][i]),
                "source":               "open-meteo-forecast",
            })

        logger.info(
            "Forecast weather fetched for (%.4f, %.4f): %d hourly slots",
            lat, lon, len(slots)
        )
        return slots

    except Exception as e:
        logger.warning(
            "Forecast weather fetch failed for (%.4f, %.4f): %s — using fallback",
            lat, lon, e
        )
        # Return 24 identical fallback slots spaced 1 hour apart
        base = datetime.now()
        return [
            {
                "hour_dt":              base.replace(minute=0, second=0, microsecond=0),
                "temperature_2m":       25.0,
                "relative_humidity_2m": 50.0,
                "precipitation":        0.0,
                "cloudcover":           40.0,
                "windspeed_10m":        5.0,
                "source":               "fallback",
            }
            for _ in range(24)
        ]


def build_ensemble_features_at(lat: float, lon: float, weather: dict, dt: datetime) -> list:
    """
    Variant of build_ensemble_features that uses a given datetime
    instead of datetime.now() — needed for forecasting future hours.
    """
    month = dt.month
    day   = dt.day
    year  = dt.year

    FFMC = DMC = DC = ISI = area = 0

    temp = weather["temperature_2m"]
    RH   = weather["relative_humidity_2m"]
    wind = weather["windspeed_10m"]
    rain = weather["precipitation"]

    temp_RH         = temp * RH
    wind_ISI        = wind * ISI
    rain_DMC        = rain * DMC
    FFMC_ISI        = FFMC * ISI
    temp_RH_ratio   = temp / (RH + 1)
    wind_rain_ratio = wind / (rain + 1)
    FFMC_sq         = FFMC ** 2
    temp_sq         = temp ** 2
    wind_sq         = wind ** 2
    ISI_log         = math.log(ISI + 1)
    area_log        = math.log(area + 1)

    features = [
        lon, lat, month, day, FFMC, DMC, DC, ISI,
        temp, RH, wind, rain, area,
        month, day, year, month * 31 + day,
        temp_RH, wind_ISI, rain_DMC, FFMC_ISI,
        temp_RH_ratio, wind_rain_ratio,
        FFMC_sq, temp_sq, wind_sq, ISI_log, area_log,
    ]
    if len(features) < ENSEMBLE_FEATURE_COUNT:
        features.extend([0] * (ENSEMBLE_FEATURE_COUNT - len(features)))

    assert len(features) == ENSEMBLE_FEATURE_COUNT, (
        f"Ensemble forecast feature count mismatch: expected {ENSEMBLE_FEATURE_COUNT}, got {len(features)}"
    )
    return features


def build_regression_features_at(lat: float, lon: float, weather: dict, sat: dict, dt: datetime) -> list:
    """
    Variant of build_regression_features that uses a given datetime
    instead of datetime.now() — needed for forecasting future hours.
    """
    lat_abs     = abs(lat)
    lon_abs     = abs(lon)
    is_northern = 1 if lat >= 0 else 0
    is_eastern  = 1 if lon >= 0 else 0

    zone_equatorial = 1 if abs(lat) < 15 else 0
    zone_temperate  = 1 if 15 <= abs(lat) < 45 else 0
    zone_polar      = 1 if abs(lat) >= 45 else 0

    month = dt.month
    day   = dt.day
    hour  = dt.hour

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin   = math.sin(2 * math.pi * day   / 31)
    day_cos   = math.cos(2 * math.pi * day   / 31)
    hour_sin  = math.sin(2 * math.pi * hour  / 24)
    hour_cos  = math.cos(2 * math.pi * hour  / 24)

    temp     = weather["temperature_2m"]
    humidity = weather["relative_humidity_2m"]
    precip   = weather["precipitation"]
    wind     = weather["windspeed_10m"]
    cloud    = weather["cloudcover"]

    dryness_index   = temp - humidity / 5
    wind_precip_idx = wind / (1 + precip)
    temp_cloud_diff = temp - cloud / 10

    brightness = sat["brightness"]
    bright_t31 = sat["bright_t31"]
    confidence = sat["confidence"]

    features = [
        lat, lon, lat_abs, lon_abs, is_northern, is_eastern,
        zone_equatorial, zone_temperate, zone_polar,
        month_sin, month_cos, day_sin, day_cos,
        hour_sin, hour_cos, temp, humidity,
        precip, wind, cloud, dryness_index, wind_precip_idx,
        temp_cloud_diff, brightness, bright_t31, confidence,
    ]

    assert len(features) == REGRESSION_FEATURE_COUNT, (
        f"Regression forecast feature count mismatch: expected {REGRESSION_FEATURE_COUNT}, got {len(features)}"
    )
    return features


# ===========================================================
#  Forecast Route — 24-hour ahead predictions
# ===========================================================
@app.route("/forecast", methods=["POST"])
@login_required
@limiter.limit("20 per minute")
def forecast():
    try:
        data = request.get_json()
        if not data or "lat" not in data or "lon" not in data:
            return jsonify({"error": "lat and lon are required"}), 400

        lat = float(data["lat"])
        lon = float(data["lon"])

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "lat/lon out of valid range"}), 400

        hourly_weather = get_forecast_weather(lat, lon)
        sat            = get_satellite_defaults()
        results        = []

        for slot in hourly_weather:
            dt      = slot["hour_dt"]
            weather = slot  # same keys as get_weather() output

            try:
                X_ens      = build_ensemble_features_at(lat, lon, weather, dt)
                X_reg      = build_regression_features_at(lat, lon, weather, sat, dt)
                fire_class = int(voting_model.predict([X_ens])[0])
                X_reg_sc   = reg_scaler.transform([X_reg])
                frp_pred   = float(reg_model.predict(X_reg_sc)[0])
            except AssertionError as ae:
                logger.error("Forecast feature assertion at %s: %s", dt.isoformat(), ae)
                continue

            results.append({
                "timestamp":     dt.isoformat(),
                "hour_label":    dt.strftime("%d %b %H:%M"),
                "fire_class":    fire_class,
                "predicted_frp": round(frp_pred, 2),
                "temperature":   round(weather["temperature_2m"], 1),
                "humidity":      round(weather["relative_humidity_2m"], 1),
                "wind":          round(weather["windspeed_10m"], 1),
                "rain":          round(weather["precipitation"], 2),
            })

        logger.info(
            "Forecast completed for user %s at (%.4f, %.4f): %d slots",
            session["user_id"], lat, lon, len(results)
        )
        return jsonify(results)

    except Exception as e:
        logger.error("Forecast error: %s", e)
        return jsonify({"error": str(e)}), 400


# ===========================================================
#  Auth Routes
# ===========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        fullname  = request.form.get("fullname", "").strip()
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        otp_input = request.form.get("otp", "").strip()

        # Validate OTP first
        otp_valid, otp_error = verify_otp(email, otp_input)
        if not otp_valid:
            flash(otp_error, "error")
            return redirect(url_for("signup"))

        if not valid_email(email):
            flash("Invalid email format.", "warning")
            return redirect(url_for("signup"))

        if not strong_password(password):
            flash(
                "Password too weak. Need: 8+ chars, uppercase, lowercase, digit, symbol (@$!%*?&).",
                "warning",
            )
            return redirect(url_for("signup"))

        password_hash = generate_password_hash(password)

        try:
            with get_db() as (conn, cursor):
                cursor.execute(
                    "INSERT INTO users (fullname, email, password_hash) VALUES (?, ?, ?)",
                    (fullname, email, password_hash),
                )
                user_id = cursor.lastrowid

            session["user_id"] = user_id
            session["email"]   = email
            flash("Signup successful! Welcome to your dashboard.", "success")
            return redirect(url_for("dashboard"))

        except sqlite3.IntegrityError:
            flash("Email already registered. Please log in.", "danger")
            return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        with get_db() as (conn, cursor):
            cursor.execute(
                "SELECT id, password_hash FROM users WHERE email = ?", (email,)
            )
            user = cursor.fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["email"]   = email
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
#  Protected Routes
# ===========================================================
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/history")
@login_required
def history():
    return render_template("history.html")


@app.route("/get_history")
@login_required
def get_history():
    user_id = session["user_id"]
    try:
        with get_db() as (conn, cursor):
            cursor.execute("""
                SELECT id, latitude, longitude, fire_class,
                       predicted_frp, weather_json, timestamp
                FROM   predictions
                WHERE  user_id = ?
                ORDER  BY timestamp DESC
            """, (user_id,))
            rows = cursor.fetchall()

        data = []
        for r in rows:
            # weather_json is stored as proper JSON — parse it back safely
            try:
                weather = json.loads(r["weather_json"]) if r["weather_json"] else {}
            except (json.JSONDecodeError, TypeError):
                weather = {}

            data.append({
                "id":            r["id"],
                "latitude":      r["latitude"],
                "longitude":     r["longitude"],
                "fire_class":    int(r["fire_class"]),
                "predicted_frp": float(r["predicted_frp"]),
                "weather":       weather,
                "timestamp":     r["timestamp"],
            })

        return jsonify(data)

    except Exception as e:
        logger.error("get_history error for user %s: %s", user_id, e)
        return jsonify([])

# ===========================================================
#  Predict Route — rate limited
# ===========================================================
@app.route("/predict", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def predict():
    try:
        data = request.get_json()
        if not data or "lat" not in data or "lon" not in data:
            return jsonify({"error": "lat and lon are required"}), 400

        lat = float(data["lat"])
        lon = float(data["lon"])

        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "lat/lon out of valid range"}), 400

        weather = get_weather(lat, lon)
        sat     = get_satellite_defaults()

        X_ens = build_ensemble_features(lat, lon, weather)
        X_reg = build_regression_features(lat, lon, weather, sat)

        fire_class = int(voting_model.predict([X_ens])[0])
        fire_label = "Fire" if fire_class == 1 else "No Fire"

        X_reg_scaled = reg_scaler.transform([X_reg])
        frp_pred     = float(reg_model.predict(X_reg_scaled)[0])

        # Persist prediction — use json.dumps so weather is real JSON
        try:
            user_id = session["user_id"]
            with get_db() as (conn, cursor):
                cursor.execute("""
                    INSERT INTO predictions
                        (user_id, latitude, longitude, fire_class, predicted_frp, weather_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    lat,
                    lon,
                    fire_class,
                    frp_pred,
                    json.dumps(weather),                        # ← proper JSON, not str()
                    datetime.now(timezone.utc).isoformat(),
                ))
            logger.info("Prediction saved for user %s at (%.4f, %.4f)", user_id, lat, lon)

        except Exception as db_e:
            # DB failure must not block the prediction response
            logger.error("Prediction DB write error for user %s: %s", user_id, db_e)

        # --- check alert settings and send email if warranted ---
        try:
            risk_label = risk_label_from_prediction(fire_class, frp_pred)
            with get_db() as (conn, cursor):
                cursor.execute(
                    "SELECT enabled, threshold FROM alert_settings WHERE user_id = ?",
                    (user_id,)
                )
                alert_row = cursor.fetchone()
                cursor.execute(
                    "SELECT email FROM users WHERE id = ?", (user_id,)
                )
                user_row = cursor.fetchone()

            if (alert_row and bool(alert_row["enabled"])
                    and user_row
                    and should_alert(risk_label, alert_row["threshold"])):
                send_alert_email(
                    user_row["email"], lat, lon, risk_label, frp_pred, weather
                )
        except Exception as alert_e:
            logger.error("Alert check error for user %s: %s", user_id, alert_e)

        return jsonify({
            "fire_label":    fire_label,
            "fire_class":    fire_class,
            "predicted_frp": round(frp_pred, 2),
            "weather":       weather,
        })

    except AssertionError as ae:
        logger.error("Feature vector assertion failed: %s", ae)
        return jsonify({"error": "Internal feature error. Contact support."}), 500

    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 400

# ===========================================================
#  Explainable AI — feature importance from both models
# ===========================================================

# Human-readable names for the regression model's 26 features
# Must match the order in build_regression_features exactly
REGRESSION_FEATURE_NAMES = [
    "Latitude", "Longitude", "Abs Latitude", "Abs Longitude",
    "Northern Hemisphere", "Eastern Hemisphere",
    "Equatorial Zone", "Temperate Zone", "Polar Zone",
    "Month (sin)", "Month (cos)", "Day (sin)", "Day (cos)",
    "Hour (sin)", "Hour (cos)",
    "Temperature (°C)", "Humidity (%)",
    "Precipitation (mm)", "Wind Speed (km/h)", "Cloud Cover (%)",
    "Dryness Index", "Wind/Precip Index", "Temp/Cloud Diff",
    "Satellite Brightness", "Satellite Bright T31", "Satellite Confidence",
]

# Human-readable names for the ensemble model's 44 features
# Must match the order in build_ensemble_features exactly
ENSEMBLE_FEATURE_NAMES = [
    "Longitude", "Latitude", "Month", "Day",
    "FFMC", "DMC", "DC", "ISI",
    "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)",
    "Precipitation (mm)", "Area",
    "Month (dup)", "Day (dup)", "Year", "Day of Year",
    "Temp × Humidity", "Wind × ISI", "Rain × DMC", "FFMC × ISI",
    "Temp/Humidity Ratio", "Wind/Rain Ratio",
    "FFMC²", "Temp²", "Wind²", "ISI (log)", "Area (log)",
] + [f"Padding_{i}" for i in range(44 - 28)]


def get_regression_importances(X_reg: list) -> list[dict]:
    importances = [float(x) for x in reg_model.feature_importances_]
    paired = list(zip(REGRESSION_FEATURE_NAMES, importances))
    paired.sort(key=lambda x: abs(x[1]), reverse=True)
    top = paired[:8]
    total = sum(abs(v) for _, v in top) or 1.0
    return [
        {
            "feature":    name,
            "importance": round(float(val), 6),
            "pct":        float(round(abs(val) / total * 100, 1)),
        }
        for name, val in top
    ]


def get_ensemble_importances() -> list[dict]:
    rf_names = ["RandomForest_SMOTE", "RandomForest_UnderSample", "RandomForest_SMOTE_Tomek"]
    all_imps = []

    for name in rf_names:
        try:
            est = voting_model.named_estimators_[name]
            all_imps.append(est.feature_importances_)
        except Exception as e:
            logger.warning("Could not read importances from %s: %s", name, e)

    if not all_imps:
        return []

    avg_imp = [float(x) for x in np.mean(all_imps, axis=0)]
    names   = ENSEMBLE_FEATURE_NAMES[:len(avg_imp)]
    paired  = list(zip(names, avg_imp))
    paired.sort(key=lambda x: abs(x[1]), reverse=True)
    top     = paired[:8]
    total   = sum(abs(v) for _, v in top) or 1.0

    return [
        {
            "feature":    name,
            "importance": round(float(val), 6),
            "pct":        float(round(abs(val) / total * 100, 1)),
        }
        for name, val in top
    ]


@app.route("/explain", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def explain():
    try:
        data = request.get_json()
        if not data or "lat" not in data or "lon" not in data:
            return jsonify({"error": "lat and lon are required"}), 400

        lat = float(data["lat"])
        lon = float(data["lon"])

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "lat/lon out of valid range"}), 400

        weather = get_weather(lat, lon)
        sat     = get_satellite_defaults()
        X_reg   = build_regression_features(lat, lon, weather, sat)

        reg_imp = get_regression_importances(X_reg)
        ens_imp = get_ensemble_importances()

        logger.info(
            "Explain request for user %s at (%.4f, %.4f)",
            session["user_id"], lat, lon
        )

        return jsonify({
            "regression_importances": reg_imp,
            "ensemble_importances":   ens_imp,
        })

    except Exception as e:
        import traceback
        logger.error("Explain error: %s\n%s", e, traceback.format_exc())  # ← ADD THIS
        return jsonify({"error": str(e)}), 400
    
# ===========================================================
#  Alert Settings — optional email alerts per user
# ===========================================================

def init_alert_settings():
    """Add alert_settings table if it doesn't exist yet."""
    with get_db() as (conn, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_settings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER UNIQUE NOT NULL,
                enabled     INTEGER NOT NULL DEFAULT 0,
                threshold   TEXT    NOT NULL DEFAULT 'Large',
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

init_alert_settings()

RISK_ORDER = ["Small Fire", "Moderate", "Large", "Extreme"]

def risk_label_from_prediction(fire_class: int, frp: float) -> str:
    if fire_class == 0:    return "No Fire Risk"
    if frp < 50:           return "Small Fire"
    if frp < 500:          return "Moderate"
    if frp < 2000:         return "Large"
    return "Extreme"

def should_alert(risk_label: str, threshold: str) -> bool:
    """Return True if risk_label meets or exceeds the user's threshold."""
    if risk_label == "No Fire Risk":
        return False
    if risk_label not in RISK_ORDER or threshold not in RISK_ORDER:
        return False
    return RISK_ORDER.index(risk_label) >= RISK_ORDER.index(threshold)

def send_alert_email(to_email: str, lat: float, lon: float,
                     risk_label: str, frp: float, weather: dict) -> bool:
    """Send a fire risk alert email to the user."""
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"⚠️ Wildfire Alert: {risk_label} Detected"
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_email

        body = f"""\
⚠️  WILDFIRE RISK ALERT

A prediction for the location you monitored has returned a high risk result.

Location  : Lat {lat:.4f}, Lon {lon:.4f}
Risk Level: {risk_label}
Predicted FRP: {frp:.2f} MW

Current Weather at that location:
  Temperature : {weather.get('temperature_2m', 'N/A')} °C
  Humidity    : {weather.get('relative_humidity_2m', 'N/A')} %
  Wind Speed  : {weather.get('windspeed_10m', 'N/A')} km/h
  Precipitation: {weather.get('precipitation', 'N/A')} mm

Please check the dashboard for more details.

— Wildfire Predictor Team

To stop receiving these alerts, log in and visit Settings → Alerts.
"""
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        logger.info("Alert email sent to %s for risk=%s", to_email, risk_label)
        return True

    except Exception as e:
        logger.error("Failed to send alert email to %s: %s", to_email, e)
        return False


@app.route("/alerts", methods=["GET"])
@login_required
def alerts_page():
    return render_template("alerts.html")


@app.route("/get_alert_settings", methods=["GET"])
@login_required
def get_alert_settings():
    user_id = session["user_id"]
    with get_db() as (conn, cursor):
        cursor.execute(
            "SELECT enabled, threshold FROM alert_settings WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()

    if row:
        return jsonify({"enabled": bool(row["enabled"]), "threshold": row["threshold"]})
    # Return defaults if no row yet
    return jsonify({"enabled": False, "threshold": "Large"})


@app.route("/save_alert_settings", methods=["POST"])
@login_required
def save_alert_settings():
    user_id = session["user_id"]
    data    = request.get_json()

    enabled   = 1 if data.get("enabled") else 0
    threshold = data.get("threshold", "Large")

    if threshold not in RISK_ORDER:
        return jsonify({"error": "Invalid threshold value"}), 400

    with get_db() as (conn, cursor):
        cursor.execute("""
            INSERT INTO alert_settings (user_id, enabled, threshold)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                enabled   = excluded.enabled,
                threshold = excluded.threshold
        """, (user_id, enabled, threshold))

    logger.info(
        "Alert settings saved for user %s: enabled=%s threshold=%s",
        user_id, bool(enabled), threshold
    )
    return jsonify({"message": "Settings saved."})

@app.route("/forgot-password", methods=["GET"])
def forgot_password():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("forgot_password.html")


@app.route("/reset-password", methods=["POST"])
def reset_password():
    try:
        data     = request.get_json()
        email    = (data or {}).get("email", "").strip()
        otp_input= (data or {}).get("otp", "").strip()
        password = (data or {}).get("password", "")

        if not email or not otp_input or not password:
            return jsonify({"error": "All fields are required."}), 400

        otp_valid, otp_error = verify_otp(email, otp_input)
        if not otp_valid:
            return jsonify({"error": otp_error}), 400

        if not strong_password(password):
            return jsonify({"error": "Password too weak. Need: 8+ chars, uppercase, lowercase, digit, symbol (@$!%*?&)."}), 400

        with get_db() as (conn, cursor):
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

        if not user:
            return jsonify({"error": "No account found with that email."}), 404

        with get_db() as (conn, cursor):
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (generate_password_hash(password), email)
            )

        logger.info("Password reset for %s", email)
        return jsonify({"message": "Password reset successful. You can now log in."})

    except Exception as e:
        logger.error("reset-password error: %s", e)
        return jsonify({"error": "Internal server error."}), 500

# ===========================================================
#  Run
# ===========================================================
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)