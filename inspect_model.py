import joblib

def inspect_pickle(path):
    print(f"\n=== Inspecting {path} ===")
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict):
            print("Type: dict")
            print("Keys:", list(obj.keys()))
            if "features" in obj:
                print(f"Feature count: {len(obj['features'])}")
                print("First few features:", obj["features"][:10])
            if "scaler" in obj:
                print("Scaler type:", type(obj["scaler"]))
            if "model" in obj:
                print("Model type:", type(obj["model"]))
        else:
            print("Type:", type(obj))
            # Try generic attributes
            if hasattr(obj, "feature_names_in_"):
                print("Feature names from model:", obj.feature_names_in_)
            elif hasattr(obj, "get_booster"):
                print("Booster feature names (XGBoost):", obj.get_booster().feature_names)
            else:
                print("No explicit feature list found.")
    except Exception as e:
        print("Error reading pickle:", e)

# === Paths ===
inspect_pickle("fire_regression_model_2.pkl")
inspect_pickle("voting_ensemble_fire.joblib")
