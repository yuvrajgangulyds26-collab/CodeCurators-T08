import joblib
import numpy as np

model = joblib.load("fire_regression_model.pkl")

print("Loaded model:", type(model))
print("Has scaler?", hasattr(model, 'named_steps'))

x = np.random.rand(1, 26)
y = model.predict(x)
print("Prediction on random 26 features:", y)
