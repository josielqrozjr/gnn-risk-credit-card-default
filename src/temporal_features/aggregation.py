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
    categorical_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate monthly records into deterministic customer-level features."""
    value_cols = list(value_cols)
    categorical_cols = list(categorical_cols)
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

    required = {id_col, time_col, *value_cols, *categorical_cols}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot aggregate missing columns: {missing}")

    features = (
        frame.sort([id_col, time_col])
        .group_by(id_col)
        .agg(
            *[pl.col(c).mean().alias(f"{c}_mean") for c in value_cols],
            *[pl.col(c).std().fill_null(0.0).alias(f"{c}_std") for c in value_cols],
            *[pl.col(c).min().alias(f"{c}_min") for c in value_cols],
            *[pl.col(c).max().alias(f"{c}_max") for c in value_cols],
            *[pl.col(c).last().alias(f"{c}_last") for c in value_cols],
            *[(pl.col(c).last() - pl.col(c).first()).alias(f"{c}_diff") for c in value_cols],
            *[
                pl.col(c).cast(pl.String).last().alias(f"{c}_last")
                for c in categorical_cols
            ],
        )
        .sort(id_col)
    )

    return features.to_pandas()
