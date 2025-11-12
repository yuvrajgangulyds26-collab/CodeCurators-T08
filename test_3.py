import joblib
import numpy as np

print("🔹 Loading voting ensemble model...")
model = joblib.load("voting_ensemble_fire.joblib")
print("✅ Model loaded successfully!")

# Try to print expected input length
try:
    n_features = model.estimators_[0].n_features_in_
except Exception:
    n_features = 44  # fallback from your error log

print(f"🧠 Model expects {n_features} features as input.\n")

# --- Create synthetic data ---
# Random but realistic values
sample = np.array([[
    7.0, 5.0, 8, 15, 85.3, 26.2, 94.3, 5.1,  # base values
    25.5, 50.2, 3.6, 0.2, 10.5, 8, 15,       # core numeric
    2024, 150,                                # year/date type
    1275.0, 18.4, 24.3,                       # derived values
    430.5, 0.51, 0.12,                        # ratios
    0.3, 650.0, 12.2, 0.05,                   # sq/log
    5.5, 3.4, 1, 0.8, 12.0, 9.5,              # filler floats
    0.6, 3.2, 4.7, 10.8, 7.1, 2.3, 0.9, 0.5,  # remaining placeholders
    0.2, 0.01, 1                              # final features
]])

print("🧾 Synthetic input shape:", sample.shape)
print("📊 First few values:", sample[0][:10])

# --- Prediction ---
try:
    pred = model.predict(sample)
    print("\n🔥 Prediction successful!")
    print("Predicted fire class:", pred[0])
except Exception as e:
    print("\n❌ Prediction failed:", e)
