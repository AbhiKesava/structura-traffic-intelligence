"""
FastAPI integration tests using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app, state
from src.data.loader import load_metr_la

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_api_state():
    """Ensure API state is initialized."""
    speed_df, adj_mx, sensor_df = load_metr_la()
    state["speed_df"] = speed_df
    state["adj_mx"] = adj_mx
    state["sensor_df"] = sensor_df


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_available" in data
    assert data["horizons_minutes"] == [15, 30, 60]


def test_api_sensors():
    response = client.get("/sensors?limit=10")
    assert response.status_code == 200
    sensors = response.json()
    assert len(sensors) == 10
    first = sensors[0]
    assert "sensor_id" in first
    assert "latitude" in first
    assert "longitude" in first
    assert "highway" in first


def test_api_predict_sensor():
    # Fetch first sensor ID
    first_sensor = str(state["sensor_df"].iloc[0]["sensor_id"])
    payload = {
        "sensor_id": first_sensor,
        "model_name": "persistence"
    }
    response = client.post("/predict/sensor", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["sensor_id"] == first_sensor
    assert len(res["predictions"]) == 3
    horizons = [p["horizon_minutes"] for p in res["predictions"]]
    assert horizons == [15, 30, 60]
    for p in res["predictions"]:
        assert p["congestion_class"] in ["Free", "Moderate", "Heavy", "Severe"]


def test_api_predict_sensor_not_found():
    payload = {
        "sensor_id": "invalid_sensor_999999",
        "model_name": "persistence"
    }
    response = client.post("/predict/sensor", json=payload)
    assert response.status_code == 404


def test_api_predict_network():
    payload = {
        "model_name": "persistence"
    }
    response = client.post("/predict/network", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_sensors"] == len(state["sensor_df"])
    assert len(data["predictions"]) == len(state["sensor_df"])
    first_item = data["predictions"][0]
    assert "speed_15m" in first_item
    assert "speed_30m" in first_item
    assert "speed_60m" in first_item
