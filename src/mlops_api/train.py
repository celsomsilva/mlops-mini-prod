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
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

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

# ---- 3. Train model (Ridge) ----
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ---- 4. Evaluate ----
preds = model.predict(X_test)
rmse = root_mean_squared_error(y_test, preds)

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

print("Model trained successfully")
print(f"RMSE: {rmse:.2f}")

