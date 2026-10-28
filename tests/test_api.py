from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["model_ready"] is True

def test_prediction():
    payload = {
        "age": 55.0,
        "trestbps": 130.0,
        "chol": 240.0,
        "thalach": 150.0,
        "oldpeak": 1.5
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert data["result"] in ["Heart Disease Detected", "Normal / No Disease"]

def test_prediction_rejects_out_of_range_values():
    payload = {
        "age": 0,
        "trestbps": 130,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.5
    }

    res = client.post("/predict", json=payload)

    assert res.status_code == 422

def test_prediction_rejects_missing_features():
    res = client.post("/predict", json={"age": 55})

    assert res.status_code == 422
