"""FastAPI entry point for the India House Price Predictor."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.prediction import router as prediction_router
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.prediction import HealthResponse
from backend.app.services.inference import InferenceService
from backend.app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory with an injectable configuration for tests."""

    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading trusted local model artifacts")
        application.state.inference = InferenceService.load(
            resolved.model_path,
            resolved.locations_path,
            resolved.metadata_path,
        )
        logger.info("Model artifacts loaded")
        yield
        application.state.inference = None

    application = FastAPI(
        title=resolved.app_name,
        version="1.0.0",
        description=(
            "Educational Indian house-price estimates. Predictions are informational and are not "
            "professional appraisals or investment advice."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    application.include_router(prediction_router)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health(request: Request) -> HealthResponse:
        service = request.app.state.inference
        return HealthResponse(
            model_loaded=isinstance(service, InferenceService),
            model_version=service.model_version,
        )

    return application


app = create_app()
