from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import math

app = Flask(__name__)
CORS(app)

# Load models
print(" Loading models...")
voting_model = joblib.load("voting_ensemble_fire.joblib")
reg_data = joblib.load("fire_regression_model.pkl")
reg_model = reg_data["model"]
reg_scaler = reg_data["scaler"]
reg_features = reg_data["features"]
print(" Models loaded successfully!")
print(f"Regression model expects {len(reg_features)} features:\n{reg_features}\n")


def make_features(data):
    """Construct full input for both models."""
    X = float(data["X"])
    Y = float(data["Y"])
    month = int(data["month"])
    day = int(data["day"])
    FFMC = float(data["FFMC"])
    DMC = float(data["DMC"])
    DC = float(data["DC"])
    ISI = float(data["ISI"])
    temp = float(data["temp"])
    RH = float(data["RH"])
    wind = float(data["wind"])
    rain = float(data["rain"])
    area = float(data["area"])

    # Derived features (just a few for illustration)
    temp_RH = temp * RH
    ISI_log = math.log(ISI + 1)
    area_log = math.log(area + 1)

    X_input = np.array([[X, Y, month, day, FFMC, DMC, DC, ISI,
                         temp, RH, wind, rain, area, temp_RH, ISI_log, area_log]])
    return X_input


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("\n Incoming data:", data)

        # For ensemble
        X_input = make_features(data)
        fire_class = int(voting_model.predict(X_input)[0])
        fire_label = " Fire" if fire_class == 1 else " No Fire"

        # For regression
        # Build dataframe with correct columns
        reg_input_dict = {f: data.get(f, None) for f in reg_features}
        print(" Raw regression input dict:")
        print(reg_input_dict)

        reg_df = pd.DataFrame([reg_input_dict])
        print(" DataFrame before scaling:")
        print(reg_df)

        reg_scaled = reg_scaler.transform(reg_df)
        print(" Scaled input sample (first row):")
        print(reg_scaled[0])

        frp_pred = float(reg_model.predict(reg_scaled)[0])
        print(f" FRP Prediction: {frp_pred:.4f}")

        return jsonify({
            "fire_class": fire_class,
            "fire_label": fire_label,
            "predicted_frp": round(frp_pred, 3)
        })

    except Exception as e:
        print(" Error:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
