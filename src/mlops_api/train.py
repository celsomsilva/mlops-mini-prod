"""
Offline training script.

Generates a trained Ridge regression model and saves the model artifact
and metadata to the models/ directory.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---- 1. Synthetic but realistic dataset ----
np.random.seed(42)

n = 1000
data = pd.DataFrame({
    "price": np.random.uniform(5, 20, n),
    "promotion": np.random.binomial(1, 0.3, n),
    "temperature": np.random.uniform(10, 35, n),
})

data["weekly_sales"] = (
    200
    - 8 * data["price"]
    + 50 * data["promotion"]
    + 2 * data["temperature"]
    + np.random.normal(0, 10, n)
)

X = data[["price", "promotion", "temperature"]]
y = data["weekly_sales"]

# ---- 2. Train / test split ----
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

logger.info("Starting model training")

# ---- 3. Train model (Ridge) ----
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ---- 4. Evaluate ----
preds = model.predict(X_test)
rmse = root_mean_squared_error(y_test, preds)
logger.info(f"Training completed | RMSE: {rmse:.2f}")

# ---- 5. Save artifacts ----
joblib.dump(model, MODEL_DIR / "model.joblib")

metadata = {
    "trained_at": datetime.utcnow().isoformat(),
    "model_type": "Ridge",
    "features": list(X.columns),
    "target": "weekly_sales",
    "rmse": rmse,
    "train_size": len(X_train),
    "test_size": len(X_test),
}

(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

logger.info("Model artifacts saved to models/")

print("Model trained successfully")
print(f"RMSE: {rmse:.2f}")

