"""Data parsing and preprocessing helpers for the house price model."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RANDOM_SEED = 42
TARGET_COLUMN = "price_rupees"

NUMERIC_FEATURES = [
    "carpet_area_sqft",
    "floor_num",
    "total_floors",
    "bathroom",
    "balcony",
    "parking",
]

LOCATION_FEATURE = ["location"]

CATEGORICAL_FEATURES = [
    "furnishing",
    "transaction",
    "ownership",
    "facing",
]

FEATURE_NAMES = (
    NUMERIC_FEATURES
    + LOCATION_FEATURE
    + CATEGORICAL_FEATURES
)

AREA_UNIT_FACTORS = {
    "sqft": 1.0,
    "sqyrd": 9.0,
    "sqm": 10.7639,
    "acre": 43_560.0,
    "hectare": 107_639.104,
    "cent": 435.6,
    "marla": 272.25,
    "kanal": 5_445.0,
    "ground": 2_400.0,
}

# These units are kept as unresolved because their conversion depends
# on the geographical context of the property.
AMBIGUOUS_AREA_UNITS = {
    "bigha",
    "biswa",
    "aankadam",
}


def _is_missing(value: Any) -> bool:
    """Check whether a scalar value represents missing data."""

    return value is None or (
        isinstance(value, float) and np.isnan(value)
    )


def parse_price_rupees(value: Any) -> float:
    """Convert a property price representation into Indian rupees."""

    if _is_missing(value):
        return np.nan

    if isinstance(value, int | float | np.number):
        amount = float(value)
        return amount if np.isfinite(amount) and amount > 0 else np.nan

    text = str(value).strip().lower()

    unavailable_markers = (
        "call for price",
        "available on request",
        "price on request",
        "contact",
    )

    if not text or any(marker in text for marker in unavailable_markers):
        return np.nan

    cleaned = (
        text.replace("₹", "")
        .replace("inr", "")
        .replace("rs.", "")
        .replace("rs", "")
        .replace(",", "")
    )

    match = re.search(r"[-+]?\d*\.?\d+", cleaned)

    if match is None:
        return np.nan

    amount = float(match.group())

    if re.search(r"\b(cr|crore|crores)\b", cleaned):
        amount *= 10_000_000
    elif re.search(r"\b(lac|lacs|lakh|lakhs)\b", cleaned):
        amount *= 100_000
    elif re.search(r"\b(thousand|k)\b", cleaned):
        amount *= 1_000

    return (
        amount
        if np.isfinite(amount) and amount > 0
        else np.nan
    )


def _detect_area_unit(text: str) -> str | None:
    """Identify the area unit contained in a raw text value."""

    normalized = (
        text.lower()
        .replace(".", "")
        .replace(" ", "")
    )

    unit_patterns = (
        ("hectare", ("hectare", "hectares")),
        ("acre", ("acre", "acres")),
        ("sqyrd", ("sqyrd", "sqyard", "squareyard", "sqyd")),
        (
            "sqm",
            (
                "sqm",
                "sqmeter",
                "squaremeter",
                "sqmetre",
                "squaremetre",
            ),
        ),
        (
            "sqft",
            (
                "sqft",
                "sqfeet",
                "squarefeet",
                "squarefoot",
            ),
        ),
        ("marla", ("marla",)),
        ("kanal", ("kanal",)),
        ("ground", ("ground",)),
        ("cent", ("cent",)),
        ("bigha", ("bigha",)),
        ("biswa", ("biswa",)),
        ("aankadam", ("aankadam", "ankanam")),
    )

    for unit, aliases in unit_patterns:
        if any(alias in normalized for alias in aliases):
            return unit

    return None


def parse_area_sqft(value: Any) -> float:
    """Convert a supported property area into square feet."""

    if _is_missing(value):
        return np.nan

    if isinstance(value, int | float | np.number):
        area = float(value)
        return area if np.isfinite(area) and area > 0 else np.nan

    text = str(value).strip().lower().replace(",", "")

    match = re.search(r"\d+(?:\.\d+)?", text)

    if match is None:
        return np.nan

    amount = float(match.group())
    unit = _detect_area_unit(text)

    if unit in AMBIGUOUS_AREA_UNITS or unit is None:
        return np.nan

    converted_area = amount * AREA_UNIT_FACTORS[unit]

    return (
        converted_area
        if np.isfinite(converted_area) and converted_area > 0
        else np.nan
    )


def _parse_floor_token(token: str) -> float:
    """Convert a single floor label into a numeric representation."""

    value = token.strip().lower()

    if "basement" in value:
        return -1.0

    if value == "ground" or value.startswith("ground "):
        return 0.0

    match = re.search(r"-?\d+", value)

    return float(match.group()) if match else np.nan


def parse_floor(value: Any) -> tuple[float, float]:
    """Extract current floor and total floors from a raw floor value."""

    if _is_missing(value):
        return np.nan, np.nan

    text = str(value).strip()

    if not text:
        return np.nan, np.nan

    parts = re.split(
        r"\s+out\s+of\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    current_floor = _parse_floor_token(parts[0])

    total_floors = (
        _parse_floor_token(parts[1])
        if len(parts) == 2
        else np.nan
    )

    return current_floor, total_floors


def parse_count(value: Any) -> float:
    """Extract a numeric count from values such as bathroom or balcony data."""

    if _is_missing(value):
        return np.nan

    text = str(value)
    match = re.search(r"\d+", text)

    if match is None:
        return np.nan

    count = float(match.group())

    if ">" in text:
        count += 1

    return count


def parse_parking(value: Any) -> float:
    """Parse parking count and discard implausible values."""

    count = parse_count(value)

    if np.isnan(count) or count < 0 or count > 10:
        return np.nan

    return count


def normalize_category(value: Any) -> str | float:
    """Normalize categorical text while preserving missing values."""

    if _is_missing(value):
        return np.nan

    normalized = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    return normalized if normalized else np.nan


def clean_raw_listings(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean raw property listings and return row-removal statistics."""

    required_columns = {
        "Amount(in rupees)",
        "location",
        "Carpet Area",
        "Floor",
        "Bathroom",
        "Balcony",
        "Car Parking",
        "Furnishing",
        "Transaction",
        "Ownership",
        "facing",
    }

    missing_columns = sorted(
        required_columns.difference(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required raw columns: {missing_columns}"
        )

    counts: dict[str, int] = {
        "raw_rows": len(raw),
    }

    duplicate_columns = [
        column
        for column in raw.columns
        if column != "Index"
    ]

    source_data = raw.drop_duplicates(
        subset=duplicate_columns
    ).copy()

    counts["duplicate_source_rows"] = (
        len(raw) - len(source_data)
    )

    cleaned = pd.DataFrame(index=source_data.index)

    cleaned[TARGET_COLUMN] = source_data[
        "Amount(in rupees)"
    ].map(parse_price_rupees)

    cleaned["carpet_area_sqft"] = source_data[
        "Carpet Area"
    ].map(parse_area_sqft)

    floors = source_data["Floor"].map(parse_floor)

    cleaned["floor_num"] = floors.map(
        lambda item: item[0]
    )

    cleaned["total_floors"] = floors.map(
        lambda item: item[1]
    )

    cleaned["bathroom"] = source_data[
        "Bathroom"
    ].map(parse_count)

    cleaned["balcony"] = source_data[
        "Balcony"
    ].map(parse_count)

    cleaned["parking"] = source_data[
        "Car Parking"
    ].map(parse_parking)

    source_mapping = {
        "location": "location",
        "furnishing": "Furnishing",
        "transaction": "Transaction",
        "ownership": "Ownership",
        "facing": "facing",
    }

    for destination, source in source_mapping.items():
        cleaned[destination] = source_data[
            source
        ].map(normalize_category)

    valid_target = (
        cleaned[TARGET_COLUMN].notna()
        & (cleaned[TARGET_COLUMN] >= 100_000)
    )

    counts["unusable_target_rows"] = int(
        (~valid_target).sum()
    )

    cleaned = cleaned.loc[valid_target].copy()

    valid_area = cleaned["carpet_area_sqft"].between(
        100,
        20_000,
        inclusive="both",
    )

    counts["missing_or_implausible_carpet_area_rows"] = int(
        (~valid_area).sum()
    )

    cleaned = cleaned.loc[valid_area].copy()

    price_per_sqft = (
        cleaned[TARGET_COLUMN]
        / cleaned["carpet_area_sqft"]
    )

    plausible_price_area = price_per_sqft.between(
        100,
        500_000,
        inclusive="both",
    )

    counts["grossly_implausible_price_area_rows"] = int(
        (~plausible_price_area).sum()
    )

    cleaned = cleaned.loc[plausible_price_area].copy()

    sale_rows = ~cleaned["transaction"].eq("Rent/Lease")

    counts["rental_rows"] = int(
        (~sale_rows).sum()
    )

    cleaned = cleaned.loc[sale_rows].copy()

    cleaned.loc[
        ~cleaned["floor_num"].between(-1, 100),
        "floor_num",
    ] = np.nan

    cleaned.loc[
        ~cleaned["total_floors"].between(1, 100),
        "total_floors",
    ] = np.nan

    cleaned.loc[
        ~cleaned["bathroom"].between(1, 11),
        "bathroom",
    ] = np.nan

    cleaned.loc[
        ~cleaned["balcony"].between(0, 11),
        "balcony",
    ] = np.nan

    cleaned = cleaned.reset_index(drop=True)

    counts["modeling_rows"] = len(cleaned)

    return (
        cleaned[[*FEATURE_NAMES, TARGET_COLUMN]],
        counts,
    )


class RareCategoryGrouper(
    BaseEstimator,
    TransformerMixin,
):
    """Group infrequent and unseen categories under a shared label."""

    def __init__(
        self,
        min_count: int = 100,
        other_label: str = "Other",
    ) -> None:
        self.min_count = min_count
        self.other_label = other_label

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> RareCategoryGrouper:
        """Identify categories that occur frequently enough to retain."""

        del y

        values = self._to_series(X)
        counts = values.value_counts()

        self.frequent_categories_ = sorted(
            counts[
                counts >= self.min_count
            ].index.tolist()
        )

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Replace rare and unseen categories with ``other_label``."""

        if not hasattr(
            self,
            "frequent_categories_",
        ):
            raise RuntimeError(
                "RareCategoryGrouper must be fitted before transform."
            )

        values = self._to_series(X)

        grouped = values.where(
            values.isin(self.frequent_categories_),
            self.other_label,
        )

        return grouped.to_numpy(
            dtype=object
        ).reshape(-1, 1)

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return the output feature name used by the transformer."""

        if input_features:
            feature_name = next(
                iter(input_features)
            )
        else:
            feature_name = "location"

        return np.asarray(
            [feature_name],
            dtype=object,
        )

    def _to_series(self, values: Any) -> pd.Series:
        """Convert supported input formats into a normalized Series."""

        if isinstance(values, pd.DataFrame):
            series = values.iloc[:, 0]

        elif isinstance(values, pd.Series):
            series = values

        else:
            array = np.asarray(
                values,
                dtype=object,
            )
            series = pd.Series(
                array.reshape(-1)
            )

        return (
            series
            .fillna(self.other_label)
            .astype(str)
            .str.strip()
            .replace(
                "",
                self.other_label,
            )
        )
