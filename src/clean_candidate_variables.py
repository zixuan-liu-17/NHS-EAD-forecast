"""Minimal data loader used by report/forecast_report.ipynb."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_KEY = "estimated_avoidable_deaths"


def load_wide_by_site(panel_path: str | Path) -> tuple[pd.DataFrame, str | None]:
    """Return a daily wide panel with one column per metric/site pair."""
    long = pd.read_parquet(panel_path)
    long["var"] = long["metric_name"].astype(str) + " @ " + long["coverage"].astype(str)
    wide = long.pivot_table(index="date", columns="var", values="value", aggfunc="mean")
    wide = wide.sort_index()

    target_cols = [c for c in wide.columns if c.startswith(TARGET_KEY + " @ ")]
    target_col = target_cols[0] if target_cols else None
    return wide, target_col
