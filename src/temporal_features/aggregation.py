"""Temporal feature aggregation utilities for customer time-series data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import polars as pl


def aggregate_temporal_features(
    df: pd.DataFrame | pl.DataFrame,
    id_col: str,
    time_col: str,
    value_cols: Iterable[str],
) -> pd.DataFrame:
    """Aggregate monthly records into static customer-level temporal features.

    For each value column this computes mean, std, last value (by time), and
    first-to-last difference.

    Returns:
        pandas.DataFrame: aggregated customer-level features.
    """
    value_cols = list(value_cols)
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

    features = (
        frame.sort([id_col, time_col])
        .group_by(id_col)
        .agg(
            *[pl.col(c).mean().alias(f"{c}_mean") for c in value_cols],
            *[pl.col(c).std().alias(f"{c}_std") for c in value_cols],
            *[pl.col(c).last().alias(f"{c}_last") for c in value_cols],
            *[(pl.col(c).last() - pl.col(c).first()).alias(f"{c}_diff") for c in value_cols],
        )
    )

    return features.to_pandas()
