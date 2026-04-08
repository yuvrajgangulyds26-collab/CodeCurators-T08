import joblib
model = joblib.load("voting_ensemble_fire.joblib")
print(model.named_estimators_.keys())