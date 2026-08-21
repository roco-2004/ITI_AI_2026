"""Tests for the dataset parsing and preprocessing utilities."""

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
    ("value", "expected"),
    [
        ("42 Lac", 4_200_000),
        ("1.2 Cr", 12_000_000),
        ("₹ 35 Lakh", 3_500_000),
        ("2 Crore", 20_000_000),
        ("4,250,000", 4_250_000),
    ],
)
def test_price_parser_handles_common_formats(
    value: str,
    expected: float,
) -> None:
    result = parse_price_rupees(value)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        "Call for Price",
        "Price on request",
        None,
        "unknown",
    ],
)
def test_price_parser_returns_nan_for_invalid_values(
    value: object,
) -> None:
    result = parse_price_rupees(value)

    assert math.isnan(result)


@pytest.mark.parametrize(
    ("value", "expected"),
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
def test_area_parser_converts_to_square_feet(
    value: str,
    expected: float,
) -> None:
    result = parse_area_sqft(value)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        "1 bigha",
        "2 biswa",
        "1 aankadam",
        "unknown",
        None,
    ],
)
def test_area_parser_rejects_ambiguous_units(
    value: object,
) -> None:
    result = parse_area_sqft(value)

    assert math.isnan(result)


@pytest.mark.parametrize(
    ("value", "expected_current", "expected_total"),
    [
        ("3 out of 10", 3.0, 10.0),
        ("Ground out of 4", 0.0, 4.0),
        ("Lower Basement out of 12", -1.0, 12.0),
        ("Upper Basement", -1.0, np.nan),
    ],
)
def test_floor_parser(
    value: str,
    expected_current: float,
    expected_total: float,
) -> None:
    current_floor, total_floors = parse_floor(value)

    assert current_floor == expected_current

    if math.isnan(expected_total):
        assert math.isnan(total_floors)
    else:
        assert total_floors == expected_total


def test_count_parser() -> None:
    assert parse_count("> 10") == 11
    assert parse_count("3") == 3


def test_parking_parser() -> None:
    assert parse_parking("2 Covered,") == 2
    assert math.isnan(parse_parking("402 Covered"))


def test_rare_categories_are_learned_from_fit_data() -> None:
    grouper = RareCategoryGrouper(min_count=2)

    training_data = pd.DataFrame(
        {
            "location": [
                "Delhi",
                "Delhi",
                "Pune",
            ]
        }
    )

    grouper.fit(training_data)

    new_data = pd.DataFrame(
        {
            "location": [
                "Delhi",
                "Pune",
                "Jaipur",
            ]
        }
    )

    transformed = grouper.transform(new_data)

    assert transformed.ravel().tolist() == [
        "Delhi",
        "Other",
        "Other",
    ]


def test_cleaning_produces_expected_schema() -> None:
    raw_data = pd.DataFrame(
        {
            "Amount(in rupees)": [
                "42 Lac",
                "Call for Price",
            ],
            "location": [
                "Delhi",
                "Delhi",
            ],
            "Carpet Area": [
                "1000 sqft",
                "900 sqft",
            ],
            "Floor": [
                "2 out of 4",
                "Ground out of 3",
            ],
            "Bathroom": [
                "2",
                "2",
            ],
            "Balcony": [
                "1",
                "1",
            ],
            "Car Parking": [
                "1 Covered",
                "1 Covered",
            ],
            "Furnishing": [
                "Semi-Furnished",
                "Unfurnished",
            ],
            "Transaction": [
                "Resale",
                "Resale",
            ],
            "Ownership": [
                "Freehold",
                "Freehold",
            ],
            "facing": [
                "East",
                "West",
            ],
        }
    )

    cleaned_data, cleaning_stats = clean_raw_listings(
        raw_data
    )

    expected_columns = [
        *FEATURE_NAMES,
        "price_rupees",
    ]

    assert cleaned_data.columns.tolist() == expected_columns
    assert len(cleaned_data) == 1
    assert cleaning_stats["unusable_target_rows"] == 1
