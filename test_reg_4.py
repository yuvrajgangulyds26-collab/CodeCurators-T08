import joblib
import numpy as np

# Load the dictionary model
data = joblib.load("fire_regression_model.pkl")

model = data["model"]
scaler = data["scaler"]
features = data["features"]

print("Model:", type(model))
print("Scaler:", type(scaler))
print("Feature count:", len(features))

# Example random test input (replace with your actual values)
x = np.random.rand(1, len(features))

# Scale and predict
x_scaled = scaler.transform(x)
y_pred = model.predict(x_scaled)

print("Prediction:", y_pred[0])
