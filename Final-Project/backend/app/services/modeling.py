"""Utilities for building regression pipelines and evaluating predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.services.preprocessing import (
    CATEGORICAL_FEATURES,
    LOCATION_FEATURE,
    NUMERIC_FEATURES,
    RareCategoryGrouper,
)


def build_preprocessor(
    location_min_count: int = 50,
) -> ColumnTransformer:
    """Create the preprocessing pipeline used by the regression models."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    location_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Other",
                ),
            ),
            (
                "rare",
                RareCategoryGrouper(
                    min_count=location_min_count
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("location", location_pipeline, LOCATION_FEATURE),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ]

    return ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    )


def build_candidates() -> dict[str, Pipeline]:
    """Build the regression models considered during model selection."""

    models: dict[str, Pipeline] = {}

    models["Dummy median"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                DummyRegressor(strategy="median"),
            ),
        ]
    )

    models["Ridge regression"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                Ridge(alpha=20.0),
            ),
        ]
    )

    gradient_boosting = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.055,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )

    models["Histogram gradient boosting log-target"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=gradient_boosting,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )

    return models


def regression_scores(
    y_true: Any,
    predictions: Any,
) -> dict[str, float]:
    """Calculate the main regression metrics on the original scale."""

    actual = np.asarray(y_true)
    predicted = np.asarray(predictions)

    return {
        "mae_rupees": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse_rupees": float(
            root_mean_squared_error(actual, predicted)
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


def learn_price_per_sqft_bounds(
    features: pd.DataFrame,
    target: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    """Estimate price-per-square-foot limits using training data."""

    price_per_area = (
        target / features["carpet_area_sqft"]
    )

    lower_bound = float(
        price_per_area.quantile(lower_quantile)
    )
    upper_bound = float(
        price_per_area.quantile(upper_quantile)
    )

    return lower_bound, upper_bound


def training_outlier_mask(
    features: pd.DataFrame,
    target: pd.Series,
    bounds: tuple[float, float],
) -> pd.Series:
    """Identify training observations inside the learned price range."""

    price_per_area = (
        target / features["carpet_area_sqft"]
    )

    return price_per_area.between(
        bounds[0],
        bounds[1],
        inclusive="both",
    )
