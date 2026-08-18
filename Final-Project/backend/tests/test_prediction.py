"""API and model-inference tests."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "location": "mumbai",
        "carpet_area_sqft": 1200.0,
        "floor_num": 4,
        "total_floors": 12,
        "bathroom": 2,
        "balcony": 2,
        "parking": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"]


def test_locations_endpoint(client: TestClient) -> None:
    response = client.get("/api/locations")
    assert response.status_code == 200
    locations = response.json()["locations"]
    assert "Other" in locations
    assert len(locations) >= 10


def test_successful_prediction(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200
    payload = response.json()
    assert math.isfinite(payload["predicted_price"])
    assert payload["predicted_price"] > 0
    assert payload["formatted_price"].startswith("₹")
    assert payload["currency"] == "INR"
    assert payload["model_version"]
    assert "informational" in payload["disclaimer"].lower()


def test_negative_area_is_rejected(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    valid_payload["carpet_area_sqft"] = -1.0
    assert client.post("/api/predict", json=valid_payload).status_code == 422


def test_missing_required_field_is_rejected(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    valid_payload.pop("bathroom")
    assert client.post("/api/predict", json=valid_payload).status_code == 422


def test_invalid_field_type_is_rejected(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    valid_payload["floor_num"] = "four"
    assert client.post("/api/predict", json=valid_payload).status_code == 422


def test_floor_above_total_is_rejected(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    valid_payload["floor_num"] = 13
    valid_payload["total_floors"] = 12
    assert client.post("/api/predict", json=valid_payload).status_code == 422


def test_unknown_location_maps_safely(
    client: TestClient,
    valid_payload: dict[str, object],
) -> None:
    valid_payload["location"] = "A location absent from training"
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["predicted_price"])


def test_model_load_failure_is_clear(tmp_path: Path) -> None:
    settings = Settings(
        model_path=tmp_path / "missing.pkl",
        locations_path=tmp_path / "missing-locations.json",
        metadata_path=tmp_path / "missing-metadata.json",
    )
    with (
        pytest.raises(RuntimeError, match="Required model artifact is missing"),
        TestClient(create_app(settings)),
    ):
        pass
