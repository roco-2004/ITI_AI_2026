"""Tests covering API availability, validation, and model inference."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture(scope="module")
def api_client() -> Iterator[TestClient]:
    """Create a reusable test client for the API."""

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def prediction_data() -> dict[str, object]:
    """Return a representative valid property payload."""

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


def test_health_check(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"]


def test_available_locations(api_client: TestClient) -> None:
    response = api_client.get("/api/locations")

    assert response.status_code == 200

    locations = response.json()["locations"]

    assert "Other" in locations
    assert len(locations) >= 10


def test_prediction_returns_valid_result(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 200

    result = response.json()

    assert math.isfinite(result["predicted_price"])
    assert result["predicted_price"] > 0
    assert result["formatted_price"].startswith("₹")
    assert result["currency"] == "INR"
    assert result["model_version"]
    assert "informational" in result["disclaimer"].lower()


def test_area_below_allowed_range_is_rejected(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    prediction_data["carpet_area_sqft"] = -1.0

    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 422


def test_required_property_field_cannot_be_omitted(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    prediction_data.pop("bathroom")

    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 422


def test_invalid_floor_value_is_rejected(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    prediction_data["floor_num"] = "four"

    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 422


def test_current_floor_cannot_exceed_building_height(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    prediction_data["floor_num"] = 13
    prediction_data["total_floors"] = 12

    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 422


def test_unseen_location_is_handled(
    api_client: TestClient,
    prediction_data: dict[str, object],
) -> None:
    prediction_data["location"] = (
        "A location absent from training"
    )

    response = api_client.post(
        "/api/predict",
        json=prediction_data,
    )

    assert response.status_code == 200
    assert math.isfinite(
        response.json()["predicted_price"]
    )


def test_missing_model_artifacts_raise_clear_error(
    tmp_path: Path,
) -> None:
    settings = Settings(
        model_path=tmp_path / "missing.pkl",
        locations_path=tmp_path / "missing-locations.json",
        metadata_path=tmp_path / "missing-metadata.json",
    )

    with (
        pytest.raises(
            RuntimeError,
            match="Required model artifact is missing",
        ),
        TestClient(create_app(settings)),
    ):
        pass
