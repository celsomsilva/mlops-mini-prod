from fastapi.testclient import TestClient
from mlops_api.api import app
from unittest.mock import patch

client = TestClient(app)


def test_predict():
    with patch("mlops_api.predict.joblib.load") as mock_load:
        mock_model = mock_load.return_value
        mock_model.predict.return_value = [123.4]

        response = client.post(
            "/predict",
            json={
                "price": 10,
                "promotion": 1,
                "temperature": 25,
            },
        )

        assert response.status_code == 200
        assert "prediction" in response.json()

