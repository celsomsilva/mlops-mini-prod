"""
Inference logic.

Loads the trained model lazily and performs predictions
based on incoming feature payloads.
"""

import joblib
import json
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
METADATA_PATH = Path("models/metadata.json")

_model = None
_metadata = None


def load_model():
    global _model, _metadata

    if _model is None:
        try:
           _model = joblib.load(MODEL_PATH)
           _metadata = json.loads(METADATA_PATH.read_text())
           logger.info("Model loaded into memory")
        except FileNotFoundError as e:
            raise RuntimeError(
                "Model artifacts not found. "
                "Ensure the model is trained before starting the API."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Failed to load model artifacts."
            ) from e



def predict(features: dict):
    load_model()

    x = pd.DataFrame([features])
    y = _model.predict(x)[0]

    return {
        "prediction": float(y),
        "model_version": _metadata["trained_at"],
        "rmse": _metadata["rmse"],
    }

