"""Model construction and evaluation helpers shared by the notebook and verification scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.services.preprocessing import (
    CATEGORICAL_FEATURES,
    LOCATION_FEATURE,
    NUMERIC_FEATURES,
    RareCategoryGrouper,
)


def build_preprocessor(location_min_count: int = 50) -> ColumnTransformer:
    """Build fitted-only preprocessing, including train-learned location grouping."""

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    location = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Other")),
            ("rare", RareCategoryGrouper(min_count=location_min_count)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("location", location, LOCATION_FEATURE),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_candidates() -> dict[str, Pipeline]:
    """Return three computationally practical regression approaches."""

    return {
        "Dummy median": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "Ridge regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", Ridge(alpha=20.0)),
            ]
        ),
        "Histogram gradient boosting log-target": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    TransformedTargetRegressor(
                        regressor=HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.055,
                            max_leaf_nodes=31,
                            min_samples_leaf=20,
                            l2_regularization=1.0,
                            random_state=42,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
    }


def regression_scores(y_true: Any, predictions: Any) -> dict[str, float]:
    """Return regression metrics on the original rupee scale."""

    return {
        "mae_rupees": float(mean_absolute_error(y_true, predictions)),
        "rmse_rupees": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def learn_price_per_sqft_bounds(
    features: pd.DataFrame,
    target: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    """Learn target-derived outlier bounds from training rows only."""

    price_per_sqft = target / features["carpet_area_sqft"]
    return (
        float(price_per_sqft.quantile(lower_quantile)),
        float(price_per_sqft.quantile(upper_quantile)),
    )


def training_outlier_mask(
    features: pd.DataFrame,
    target: pd.Series,
    bounds: tuple[float, float],
) -> pd.Series:
    """Apply already-learned price-per-square-foot bounds to training rows."""

    price_per_sqft = target / features["carpet_area_sqft"]
    return price_per_sqft.between(bounds[0], bounds[1], inclusive="both")
