"""Prediction and location API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.schemas.prediction import (
    LocationsResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.inference import InferenceService

router = APIRouter(prefix="/api", tags=["prediction"])


def get_inference_service(request: Request) -> InferenceService:
    """Read the process-wide service initialized by the app lifespan."""

    service = getattr(request.app.state, "inference", None)
    if not isinstance(service, InferenceService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available.",
        )
    return service


@router.get("/locations", response_model=LocationsResponse)
def locations(request: Request) -> LocationsResponse:
    service = get_inference_service(request)
    return LocationsResponse(locations=service.locations)


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service = get_inference_service(request)
    try:
        return service.predict(payload)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed safely.",
        ) from exc
