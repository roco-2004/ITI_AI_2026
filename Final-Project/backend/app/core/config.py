"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated backend configuration."""

    app_name: str = "India House Price Predictor API"
    environment: str = "development"
    log_level: str = "INFO"
    model_path: Path = PROJECT_ROOT / "models" / "house_price.pkl"
    locations_path: Path = PROJECT_ROOT / "models" / "locations.json"
    metadata_path: Path = PROJECT_ROOT / "models" / "model_metadata.json"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_prefix="HOUSE_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the process lifetime."""

    return Settings()
