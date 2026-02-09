"""
API contract tests.

These tests validate the public behavior of the FastAPI application
without depending on trained model artifacts.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch

from mlops_api.api import app

client = TestClient(app)


def test_health():
    """
    Health endpoint should respond with service status.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    """
    Prediction endpoint should accept valid input and return a prediction.

    The model is mocked to avoid coupling tests to training artifacts.
    """
    # Mock model loading to keep tests independent from filesystem state
    with patch("mlops_api.predict.joblib.load") as mock_load:
        mock_model = mock_load.return_value
        mock_model.predict.return_value = [123.4]

        response = client.post(
            "/predict",
            json={
                "price": 10.0,
                "promotion": 1,
                "temperature": 25.0,
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert "prediction" in body
        assert "model_version" in body
        assert "rmse" in body

