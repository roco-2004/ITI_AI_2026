"""Focused tests for reusable dataset preprocessing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.app.services.preprocessing import (
    FEATURE_NAMES,
    RareCategoryGrouper,
    clean_raw_listings,
    parse_area_sqft,
    parse_count,
    parse_floor,
    parse_parking,
    parse_price_rupees,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42 Lac", 4_200_000),
        ("1.2 Cr", 12_000_000),
        ("₹ 35 Lakh", 3_500_000),
        ("2 Crore", 20_000_000),
        ("4,250,000", 4_250_000),
    ],
)
def test_parse_price_rupees(raw: str, expected: float) -> None:
    assert parse_price_rupees(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["Call for Price", "Price on request", None, "unknown"])
def test_parse_price_rejects_unavailable_values(raw: object) -> None:
    assert math.isnan(parse_price_rupees(raw))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1200 sqft", 1200),
        ("140 sqm", 1506.946),
        ("100 sqyrd", 900),
        ("1 acre", 43_560),
        ("1 hectare", 107_639.104),
        ("1 cent", 435.6),
        ("1 marla", 272.25),
        ("1 kanal", 5_445),
        ("1 ground", 2_400),
    ],
)
def test_parse_area_sqft(raw: str, expected: float) -> None:
    assert parse_area_sqft(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["1 bigha", "2 biswa", "1 aankadam", "unknown", None])
def test_ambiguous_or_invalid_area_is_missing(raw: object) -> None:
    assert math.isnan(parse_area_sqft(raw))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 out of 10", (3.0, 10.0)),
        ("Ground out of 4", (0.0, 4.0)),
        ("Lower Basement out of 12", (-1.0, 12.0)),
        ("Upper Basement", (-1.0, np.nan)),
    ],
)
def test_parse_floor(raw: str, expected: tuple[float, float]) -> None:
    actual = parse_floor(raw)
    assert actual[0] == expected[0]
    if math.isnan(expected[1]):
        assert math.isnan(actual[1])
    else:
        assert actual[1] == expected[1]


def test_count_and_parking_parsing() -> None:
    assert parse_count("> 10") == 11
    assert parse_count("3") == 3
    assert parse_parking("2 Covered,") == 2
    assert math.isnan(parse_parking("402 Covered"))


def test_unknown_location_grouping_is_learned_from_training_only() -> None:
    grouper = RareCategoryGrouper(min_count=2)
    training = pd.DataFrame({"location": ["Delhi", "Delhi", "Pune"]})
    grouper.fit(training)
    transformed = grouper.transform(pd.DataFrame({"location": ["Delhi", "Pune", "Jaipur"]}))
    assert transformed.ravel().tolist() == ["Delhi", "Other", "Other"]


def test_cleaned_feature_schema_and_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "Amount(in rupees)": ["42 Lac", "Call for Price"],
            "location": ["Delhi", "Delhi"],
            "Carpet Area": ["1000 sqft", "900 sqft"],
            "Floor": ["2 out of 4", "Ground out of 3"],
            "Bathroom": ["2", "2"],
            "Balcony": ["1", "1"],
            "Car Parking": ["1 Covered", "1 Covered"],
            "Furnishing": ["Semi-Furnished", "Unfurnished"],
            "Transaction": ["Resale", "Resale"],
            "Ownership": ["Freehold", "Freehold"],
            "facing": ["East", "West"],
        }
    )
    cleaned, counts = clean_raw_listings(raw)
    assert cleaned.columns.tolist() == [*FEATURE_NAMES, "price_rupees"]
    assert len(cleaned) == 1
    assert counts["unusable_target_rows"] == 1
