"""Request and response models used by the prediction API."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Furnishing = Literal[
    "Furnished",
    "Semi-Furnished",
    "Unfurnished",
]

Transaction = Literal[
    "New Property",
    "Other",
    "Resale",
]

Ownership = Literal[
    "Co-operative Society",
    "Freehold",
    "Leasehold",
    "Power Of Attorney",
]

Facing = Literal[
    "East",
    "North",
    "North - East",
    "North - West",
    "South",
    "South - East",
    "South -West",
    "West",
]


class PredictionRequest(BaseModel):
    """Input data required to generate a house price prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    location: Annotated[
        str,
        Field(min_length=1, max_length=120),
    ]

    carpet_area_sqft: Annotated[
        float,
        Field(strict=True, ge=100, le=20_000),
    ]

    floor_num: Annotated[
        int,
        Field(strict=True, ge=-1, le=100),
    ]

    total_floors: Annotated[
        int,
        Field(strict=True, ge=1, le=100),
    ]

    bathroom: Annotated[
        int,
        Field(strict=True, ge=1, le=11),
    ]

    balcony: Annotated[
        int,
        Field(strict=True, ge=0, le=11),
    ]

    parking: Annotated[
        int,
        Field(strict=True, ge=0, le=10),
    ]

    furnishing: Furnishing
    transaction: Transaction
    ownership: Ownership
    facing: Facing

    @field_validator("carpet_area_sqft")
    @classmethod
    def validate_area(cls, value: float) -> float:
        """Make sure the supplied area is a valid finite number."""

        if not math.isfinite(value):
            raise ValueError("carpet_area_sqft must be finite")

        return value

    @model_validator(mode="after")
    def validate_floor_numbers(self) -> PredictionRequest:
        """Ensure the selected floor is within the building range."""

        if self.floor_num > self.total_floors:
            raise ValueError("floor_num cannot exceed total_floors")

        return self


class PredictionResponse(BaseModel):
    """Response returned after generating a house price estimate."""

    predicted_price: float
    formatted_price: str
    currency: Literal["INR"] = "INR"
    model_version: str
    disclaimer: str


class LocationsResponse(BaseModel):
    """List of locations available to the prediction model."""

    locations: list[str]
    other_label: str = "Other"


class HealthResponse(BaseModel):
    """Information about the API and model availability."""

    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_version: str
