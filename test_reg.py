import joblib
import numpy as np
import pandas as pd

# === Load model bundle ===
bundle_path = "fire_regression_model_2.pkl"
print(f"Loading regression bundle: {bundle_path}")
bundle = joblib.load(bundle_path)

model = bundle.get("model")
scaler = bundle.get("scaler")
feature_names = bundle.get("features")

print("\nLoaded bundle keys:", list(bundle.keys()))
print("Scaler present:", scaler is not None)
print("Number of features expected:", len(feature_names))

# === Step 1: Base random features ===
n_samples = 10
synthetic_data = pd.DataFrame({
    "lat": np.random.uniform(-60, 60, n_samples),
    "long": np.random.uniform(-180, 180, n_samples),
    "month_sin": np.random.uniform(-1, 1, n_samples),
    "month_cos": np.random.uniform(-1, 1, n_samples),
    "day_sin": np.random.uniform(-1, 1, n_samples),
    "day_cos": np.random.uniform(-1, 1, n_samples),
    "hour_sin": np.random.uniform(-1, 1, n_samples),
    "hour_cos": np.random.uniform(-1, 1, n_samples),
    "temp2m": np.random.uniform(0, 40, n_samples),
    "humidity2m": np.random.uniform(10, 90, n_samples),
    "precip": np.random.uniform(0, 20, n_samples),
    "windspeed10m": np.random.uniform(0, 40, n_samples),
    "cloudcover": np.random.uniform(0, 100, n_samples),
    "dryness_index": np.random.uniform(0, 1, n_samples),
    "wind_precip_index": np.random.uniform(0, 1, n_samples),
    "temp_cloud_diff": np.random.uniform(0, 40, n_samples),
    "brightness": np.random.uniform(100, 400, n_samples),
    "bright_t31": np.random.uniform(250, 350, n_samples),
    "confidence": np.random.randint(0, 100, n_samples),
})

# === Step 2: Derived features ===
synthetic_data["lat_abs"] = np.abs(synthetic_data["lat"])
synthetic_data["long_abs"] = np.abs(synthetic_data["long"])
synthetic_data["is_northern"] = (synthetic_data["lat"] > 0).astype(int)
synthetic_data["is_eastern"] = (synthetic_data["long"] > 0).astype(int)
synthetic_data["zone_equatorial"] = (synthetic_data["lat"].abs() < 15).astype(int)
synthetic_data["zone_temperate"] = ((synthetic_data["lat"].abs() >= 15) & (synthetic_data["lat"].abs() < 45)).astype(int)
synthetic_data["zone_polar"] = (synthetic_data["lat"].abs() >= 45).astype(int)

# === Step 3: Align with model’s feature order ===
X = synthetic_data[feature_names]

# === Step 4: Scale if needed ===
if scaler is not None:
    X_scaled = scaler.transform(X)
else:
    X_scaled = X.values

# === Step 5: Predict ===
preds = model.predict(X_scaled)
preds = np.maximum(preds, 0)  # FRP cannot be negative

# === Step 6: Display results ===
synthetic_data["predicted_frp"] = preds
print("\nPredictions:")
for i, val in enumerate(preds, 1):
    print(f" sample {i:02d} -> predicted FRP: {val:.2f}")

print(f"\nAverage predicted FRP: {np.mean(preds):.2f}")
