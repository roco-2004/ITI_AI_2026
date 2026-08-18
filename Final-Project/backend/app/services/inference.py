"""Trusted local model loading and prediction service."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
from backend.app.services.preprocessing import FEATURE_NAMES

DISCLAIMER = (
    "Informational estimate only; not a professional appraisal or investment recommendation."
)


def request_to_frame(request: PredictionRequest, allowed_locations: set[str]) -> pd.DataFrame:
    """Create a one-row frame with the exact trained feature names."""

    values = request.model_dump()
    if values["location"] not in allowed_locations:
        values["location"] = "Other"
    frame = pd.DataFrame([values], columns=FEATURE_NAMES)
    if frame.shape != (1, len(FEATURE_NAMES)):
        raise RuntimeError("Prediction frame does not match the trained feature schema.")
    return frame


def format_inr(value: float) -> str:
    """Format rupees using lakh/crore conventions."""

    if value >= 10_000_000:
        return f"₹{value / 10_000_000:,.2f} Crore"
    if value >= 100_000:
        return f"₹{value / 100_000:,.2f} Lakh"
    return f"₹{value:,.0f}"


class InferenceService:
    """Loaded model artifact and immutable inference metadata."""

    def __init__(
        self,
        model: Any,
        locations: list[str],
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.locations = sorted(locations)
        self.allowed_locations = set(locations)
        self.metadata = metadata
        self.model_version = str(metadata["model_version"])

    @classmethod
    def load(
        cls,
        model_path: Path,
        locations_path: Path,
        metadata_path: Path,
    ) -> InferenceService:
        """Load trusted local artifacts once, failing clearly on absence/incompatibility."""

        paths = (model_path, locations_path, metadata_path)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Required model artifact is missing: {missing}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            locations = json.loads(locations_path.read_text(encoding="utf-8"))
            trained_version = str(metadata["versions"]["scikit_learn"])
            if trained_version != sklearn.__version__:
                raise RuntimeError(
                    "scikit-learn version mismatch: "
                    f"artifact={trained_version}, runtime={sklearn.__version__}"
                )
            model = joblib.load(model_path)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unable to load trusted local model artifacts: {exc}") from exc

        if not isinstance(locations, list) or "Other" not in locations:
            raise RuntimeError("locations.json must contain a list including 'Other'.")
        return cls(model=model, locations=locations, metadata=metadata)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run one finite pipeline prediction."""

        frame = request_to_frame(request, self.allowed_locations)
        prediction = float(self.model.predict(frame)[0])
        if not math.isfinite(prediction) or prediction <= 0:
            raise RuntimeError("The model returned an invalid prediction.")
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            formatted_price=format_inr(prediction),
            currency="INR",
            model_version=self.model_version,
            disclaimer=DISCLAIMER,
        )
