import joblib

data = joblib.load("fire_regression_model.pkl")

print("Keys inside pickle:", data.keys())

if "model" in data:
    model = data["model"]
    print("Loaded model type:", type(model))
    print("Feature names:", data.get("feature_names", "N/A"))
