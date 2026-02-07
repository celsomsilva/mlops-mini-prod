from fastapi.testclient import TestClient
from mlops_api.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_predict():
    response = client.post(
        "/predict",
        json={
            "price": 10.0,
            "promotion": 1,
            "temperature": 22.0
        }
    )

    assert response.status_code == 200
    body = response.json()

    assert "prediction" in body
    assert "model_version" in body
    assert "rmse" in body


