"""Minimal model factory used by report/forecast_report.ipynb."""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def make_model(kind: str):
    """Return the tree model used for the severity nowcast."""
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=123,
            n_jobs=-1,
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=123,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=123,
    )
