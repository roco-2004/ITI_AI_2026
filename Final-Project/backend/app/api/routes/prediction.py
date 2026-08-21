"""API endpoints for house price predictions and available locations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.schemas.prediction import (
    LocationsResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.inference import InferenceService


router = APIRouter(
    prefix="/api",
    tags=["prediction"],
)


def get_inference_service(request: Request) -> InferenceService:
    """Return the initialized inference service for the current application."""

    inference_service = getattr(
        request.app.state,
        "inference",
        None,
    )

    if not isinstance(inference_service, InferenceService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction service is currently unavailable.",
        )

    return inference_service


@router.get(
    "/locations",
    response_model=LocationsResponse,
)
def get_locations(request: Request) -> LocationsResponse:
    """Return the locations supported by the prediction model."""

    service = get_inference_service(request)

    return LocationsResponse(
        locations=service.locations,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def create_prediction(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Generate a house price prediction from the supplied property data."""

    service = get_inference_service(request)

    try:
        prediction = service.predict(payload)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete the prediction.",
        ) from exc

    return prediction
