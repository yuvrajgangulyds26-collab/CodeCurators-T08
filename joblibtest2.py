import joblib
reg_data = joblib.load("fire_regression_model_2.pkl")
reg_model = reg_data["model"]
print(hasattr(reg_model, "feature_importances_"))
print(type(reg_model))