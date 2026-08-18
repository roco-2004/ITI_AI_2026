"""Exercise the running API through its public HTTP contract."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from typing import Any


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health_status, health = request_json(f"{base_url}/health")
    assert health_status == 200 and health["status"] == "ok" and health["model_loaded"]

    locations_status, locations = request_json(f"{base_url}/api/locations")
    assert locations_status == 200 and len(locations["locations"]) >= 2
    known_location = next(item for item in locations["locations"] if item != "Other")

    payload = {
        "location": known_location,
        "carpet_area_sqft": 1200.0,
        "floor_num": 3,
        "total_floors": 10,
        "bathroom": 2,
        "balcony": 1,
        "parking": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    prediction_status, prediction = request_json(f"{base_url}/api/predict", payload=payload)
    assert prediction_status == 200
    assert prediction["currency"] == "INR"
    assert math.isfinite(prediction["predicted_price"]) and prediction["predicted_price"] > 0

    invalid = dict(payload, carpet_area_sqft=0.0)
    invalid_status, invalid_response = request_json(f"{base_url}/api/predict", payload=invalid)
    assert invalid_status == 422 and invalid_response["detail"]

    print(
        json.dumps(
            {
                "health": health,
                "location_count": len(locations["locations"]),
                "sample_location": known_location,
                "prediction": prediction,
                "invalid_request_status": invalid_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

