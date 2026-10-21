"""Test Cover Test FastAPI application and endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert data["service"] == "remicare-cover-test-ai"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "remicare-cover-test-ai"
    assert "model" in data


def test_save_cover_test_session_endpoint(client, tmp_path, monkeypatch):
    payload = {
        "sessionId": "test-session-cover-001",
        "metadata": {
            "samplingRate": 15,
            "testType": "cover_test"
        },
        "samples": [
            {
                "timestamp": 100.0,
                "phase": "BASELINE",
                "leftEye": {"x": 0.45, "y": 0.5},
                "rightEye": {"x": 0.55, "y": 0.5}
            },
            {
                "timestamp": 166.7,
                "phase": "BASELINE",
                "leftEye": {"x": 0.45, "y": 0.5},
                "rightEye": {"x": 0.55, "y": 0.5}
            }
        ]
    }
    response = client.post("/api/cover-test/sessions", json=payload)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["sessionId"] == "test-session-cover-001"
    assert res_data["saved"] is True
    assert res_data["sampleCount"] == 2
