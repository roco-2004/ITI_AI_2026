"""Reusable parsing, cleaning, and category-grouping utilities.

Raw listings are transformed into a deliberately small, documented feature schema. The
functions are used by the executed notebook and are importable when the serialized sklearn
pipeline is loaded by FastAPI.
"""

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
CATEGORICAL_FEATURES = ["furnishing", "transaction", "ownership", "facing"]
FEATURE_NAMES = NUMERIC_FEATURES + LOCATION_FEATURE + CATEGORICAL_FEATURES

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

# Bigha, biswa, and aankadam vary by region or lack enough context in this dataset. They are
# recognized but intentionally returned as missing rather than converted with false precision.
AMBIGUOUS_AREA_UNITS = {"bigha", "biswa", "aankadam"}


def _missing(value: Any) -> bool:
    """Return True for scalar missing values without applying pd.isna to arrays."""

    return value is None or (isinstance(value, float) and np.isnan(value))


def parse_price_rupees(value: Any) -> float:
    """Parse an Indian property price into rupees; return NaN when unavailable."""

    if _missing(value):
        return np.nan
    if isinstance(value, int | float | np.number):
        numeric = float(value)
        return numeric if np.isfinite(numeric) and numeric > 0 else np.nan

    text = str(value).strip().lower()
    unavailable = ("call for price", "available on request", "price on request", "contact")
    if not text or any(marker in text for marker in unavailable):
        return np.nan

    cleaned = (
        text.replace("₹", "")
        .replace("inr", "")
        .replace("rs.", "")
        .replace("rs", "")
        .replace(",", "")
    )
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if not match:
        return np.nan

    number = float(match.group())
    if re.search(r"\b(cr|crore|crores)\b", cleaned):
        number *= 10_000_000
    elif re.search(r"\b(lac|lacs|lakh|lakhs)\b", cleaned):
        number *= 100_000
    elif re.search(r"\b(thousand|k)\b", cleaned):
        number *= 1_000
    return number if np.isfinite(number) and number > 0 else np.nan


def _detect_area_unit(text: str) -> str | None:
    normalized = text.lower().replace(".", "").replace(" ", "")
    patterns = (
        ("hectare", ("hectare", "hectares")),
        ("acre", ("acre", "acres")),
        ("sqyrd", ("sqyrd", "sqyard", "squareyard", "sqyd")),
        ("sqm", ("sqm", "sqmeter", "squaremeter", "sqmetre", "squaremetre")),
        ("sqft", ("sqft", "sqfeet", "squarefeet", "squarefoot")),
        ("marla", ("marla",)),
        ("kanal", ("kanal",)),
        ("ground", ("ground",)),
        ("cent", ("cent",)),
        ("bigha", ("bigha",)),
        ("biswa", ("biswa",)),
        ("aankadam", ("aankadam", "ankanam")),
    )
    for unit, aliases in patterns:
        if any(alias in normalized for alias in aliases):
            return unit
    return None


def parse_area_sqft(value: Any) -> float:
    """Parse a supported area value and convert it to square feet."""

    if _missing(value):
        return np.nan
    if isinstance(value, int | float | np.number):
        numeric = float(value)
        return numeric if np.isfinite(numeric) and numeric > 0 else np.nan

    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    number = float(match.group())
    unit = _detect_area_unit(text)
    if unit in AMBIGUOUS_AREA_UNITS or unit is None:
        return np.nan
    result = number * AREA_UNIT_FACTORS[unit]
    return result if np.isfinite(result) and result > 0 else np.nan


def _parse_floor_token(token: str) -> float:
    token = token.strip().lower()
    if "basement" in token:
        return -1.0
    if token == "ground" or token.startswith("ground "):
        return 0.0
    match = re.search(r"-?\d+", token)
    return float(match.group()) if match else np.nan


def parse_floor(value: Any) -> tuple[float, float]:
    """Parse current and total floors, supporting ground and basement labels."""

    if _missing(value):
        return np.nan, np.nan
    text = str(value).strip()
    if not text:
        return np.nan, np.nan
    parts = re.split(r"\s+out\s+of\s+", text, maxsplit=1, flags=re.IGNORECASE)
    current = _parse_floor_token(parts[0])
    total = _parse_floor_token(parts[1]) if len(parts) == 2 else np.nan
    return current, total


def parse_count(value: Any) -> float:
    """Parse integer-like bathroom or balcony values, including '> 10'."""

    if _missing(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    if not match:
        return np.nan
    number = float(match.group())
    if ">" in str(value):
        number += 1
    return number


def parse_parking(value: Any) -> float:
    """Parse private parking count; treat implausible communal counts as missing."""

    number = parse_count(value)
    if np.isnan(number) or number < 0 or number > 10:
        return np.nan
    return number


def normalize_category(value: Any) -> str | float:
    """Normalize categorical whitespace while retaining missing values for imputation."""

    if _missing(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else np.nan


def clean_raw_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create the modeling population and return transparent row-removal counts.

    Only deterministic format/plausibility rules are applied to the full population. Target-
    derived price-per-square-foot outlier bounds are learned later from the training split only.
    """

    required = {
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
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required raw columns: {missing_columns}")

    counts: dict[str, int] = {"raw_rows": len(raw)}
    duplicate_basis = [column for column in raw.columns if column != "Index"]
    deduplicated_raw = raw.drop_duplicates(subset=duplicate_basis).copy()
    counts["duplicate_source_rows"] = len(raw) - len(deduplicated_raw)

    cleaned = pd.DataFrame(index=deduplicated_raw.index)
    cleaned[TARGET_COLUMN] = deduplicated_raw["Amount(in rupees)"].map(parse_price_rupees)
    cleaned["carpet_area_sqft"] = deduplicated_raw["Carpet Area"].map(parse_area_sqft)

    floors = deduplicated_raw["Floor"].map(parse_floor)
    cleaned["floor_num"] = floors.map(lambda value: value[0])
    cleaned["total_floors"] = floors.map(lambda value: value[1])
    cleaned["bathroom"] = deduplicated_raw["Bathroom"].map(parse_count)
    cleaned["balcony"] = deduplicated_raw["Balcony"].map(parse_count)
    cleaned["parking"] = deduplicated_raw["Car Parking"].map(parse_parking)

    source_columns = {
        "location": "location",
        "furnishing": "Furnishing",
        "transaction": "Transaction",
        "ownership": "Ownership",
        "facing": "facing",
    }
    for output, source in source_columns.items():
        cleaned[output] = deduplicated_raw[source].map(normalize_category)

    valid_target = cleaned[TARGET_COLUMN].notna() & (cleaned[TARGET_COLUMN] >= 100_000)
    counts["unusable_target_rows"] = int((~valid_target).sum())
    cleaned = cleaned.loc[valid_target].copy()

    valid_area = cleaned["carpet_area_sqft"].between(100, 20_000, inclusive="both")
    counts["missing_or_implausible_carpet_area_rows"] = int((~valid_area).sum())
    cleaned = cleaned.loc[valid_area].copy()

    # A deliberately broad, fixed integrity rule removes only obvious unit/zero corruption.
    # Tighter 1st/99th percentile bounds are learned later from the training split alone.
    gross_price_per_sqft = cleaned[TARGET_COLUMN] / cleaned["carpet_area_sqft"]
    grossly_plausible = gross_price_per_sqft.between(100, 500_000, inclusive="both")
    counts["grossly_implausible_price_area_rows"] = int((~grossly_plausible).sum())
    cleaned = cleaned.loc[grossly_plausible].copy()

    sale_transaction = ~cleaned["transaction"].eq("Rent/Lease")
    counts["rental_rows"] = int((~sale_transaction).sum())
    cleaned = cleaned.loc[sale_transaction].copy()

    cleaned.loc[~cleaned["floor_num"].between(-1, 100), "floor_num"] = np.nan
    cleaned.loc[~cleaned["total_floors"].between(1, 100), "total_floors"] = np.nan
    cleaned.loc[~cleaned["bathroom"].between(1, 11), "bathroom"] = np.nan
    cleaned.loc[~cleaned["balcony"].between(0, 11), "balcony"] = np.nan

    cleaned = cleaned.reset_index(drop=True)
    counts["modeling_rows"] = len(cleaned)
    return cleaned[[*FEATURE_NAMES, TARGET_COLUMN]], counts


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Learn frequent values during fit and map rare/unseen values to ``Other``."""

    def __init__(self, min_count: int = 100, other_label: str = "Other") -> None:
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X: Any, y: Any = None) -> RareCategoryGrouper:
        del y
        series = self._as_series(X)
        counts = series.value_counts()
        self.frequent_categories_ = sorted(counts[counts >= self.min_count].index.tolist())
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "frequent_categories_"):
            raise RuntimeError("RareCategoryGrouper must be fitted before transform.")
        series = self._as_series(X)
        grouped = series.where(series.isin(self.frequent_categories_), self.other_label)
        return grouped.to_numpy(dtype=object).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        name = next(iter(input_features), "location") if input_features else "location"
        return np.asarray([name], dtype=object)

    def _as_series(self, values: Any) -> pd.Series:
        if isinstance(values, pd.DataFrame):
            series = values.iloc[:, 0]
        elif isinstance(values, pd.Series):
            series = values
        else:
            array = np.asarray(values, dtype=object)
            series = pd.Series(array.reshape(-1))
        return series.fillna(self.other_label).astype(str).str.strip().replace("", self.other_label)
