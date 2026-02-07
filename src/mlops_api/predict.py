import json
import joblib
import numpy as np

model = joblib.load("models/model.joblib")
metadata = json.load(open("models/metadata.json"))


def predict(features: dict):
    x = np.array([[
        features["price"],
        features["promotion"],
        features["temperature"]
    ]])

    y = model.predict(x)[0]

    return {
        "prediction": float(y),
        "model_version": metadata["trained_at"],
        "rmse": metadata["rmse"]
    }

