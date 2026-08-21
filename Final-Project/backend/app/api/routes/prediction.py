"""Routes for property predictions and supported locations."""

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


def get_inference_service(
    request: Request,
) -> InferenceService:
    """Get the inference service initialized during application startup."""

    service = getattr(
        request.app.state,
        "inference",
        None,
    )

    if not isinstance(service, InferenceService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not available.",
        )

    return service


@router.get(
    "/locations",
    response_model=LocationsResponse,
)
def get_locations(
    request: Request,
) -> LocationsResponse:
    """Provide the list of supported property locations."""

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
    """Return a price estimate for the submitted property."""

    service = get_inference_service(request)

    try:
        result = service.predict(payload)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be generated.",
        ) from exc

    return result
