"""Main FastAPI application for the house price prediction service."""

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
    """Create and configure the FastAPI application."""

    config = settings if settings is not None else get_settings()
    configure_logging(config.log_level)

    @asynccontextmanager
    async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Load the model resources when the application starts."""

        logger.info("Starting application and loading model resources")

        inference_service = InferenceService.load(
            config.model_path,
            config.locations_path,
            config.metadata_path,
        )

        application.state.inference = inference_service

        logger.info("Model resources loaded successfully")

        yield

        logger.info("Shutting down application")
        application.state.inference = None

    app_instance = FastAPI(
        title=config.app_name,
        version="1.0.0",
        description=(
            "API for estimating house prices in India using a trained "
            "machine learning model. Predictions are provided for "
            "informational and educational purposes."
        ),
        lifespan=app_lifespan,
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app_instance.include_router(prediction_router)

    @app_instance.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    def health_check(request: Request) -> HealthResponse:
        """Return the current model loading status."""

        inference = request.app.state.inference

        return HealthResponse(
            model_loaded=isinstance(inference, InferenceService),
            model_version=inference.model_version,
        )

    return app_instance


app = create_app()
