import joblib
import numpy as np

print("Loading regression model...")
reg_data = joblib.load("fire_regression_model.pkl")
reg_model = reg_data["model"]
reg_scaler = reg_data["scaler"]
reg_features = reg_data["features"]
print("Loaded regression model successfully!")
print("Regression features:", reg_features)

# Example test inputs (manually varied)
sample_1 = {
    "X": 5, "Y": 5, "month": 8, "day": 15, "FFMC": 90, "DMC": 120, "DC": 500,
    "ISI": 10, "temp": 25, "RH": 35, "wind": 5, "rain": 0.2, "area": 15
}
sample_2 = {
    "X": 3, "Y": 7, "month": 3, "day": 9, "FFMC": 70, "DMC": 45, "DC": 200,
    "ISI": 4, "temp": 10, "RH": 80, "wind": 1, "rain": 2, "area": 0.5
}

def make_vector(sample):
    # Build input array based on reg_features order
    return np.array([[sample.get(f, 0) for f in reg_features]])

# Scale and predict
for i, sample in enumerate([sample_1, sample_2], start=1):
    X_reg = make_vector(sample)
    X_scaled = reg_scaler.transform(X_reg)
    pred = reg_model.predict(X_scaled)[0]
    print(f"\nSample {i} prediction: {pred:.2f}")
