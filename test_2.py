import joblib

clf_data = joblib.load("fire_classification_model_2.pkl")
scaler = clf_data["scaler"]
print("Scaler expects features:", scaler.n_features_in_)
