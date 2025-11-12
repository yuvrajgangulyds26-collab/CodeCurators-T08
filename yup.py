import joblib

model_data = joblib.load("fire_classification_model.pkl")
model_data_1 = joblib.load("fire_regression_model.pkl")

print(type(model_data))
print(type(model_data_1))
print("Keys:", model_data.keys())
print("Features:", model_data["features"])
print("Keys:", model_data_1.keys())
print("Features:", model_data_1["features"])
