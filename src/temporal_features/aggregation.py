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
    windows: Iterable[int] = (),
) -> pd.DataFrame:
    """Aggregate monthly records into full-history or trailing-window features.

    With no windows, preserve the original full-history feature names and schema.
    When windows are provided, emit independent features for each trailing window.
    """
    value_cols = list(value_cols)
    categorical_cols = list(categorical_cols)
    window_sizes = tuple(windows)
    if any(not isinstance(window, int) or window < 1 for window in window_sizes):
        raise ValueError("windows must contain positive integers")
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError("windows must not contain duplicate sizes")
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

    required = {id_col, time_col, *value_cols, *categorical_cols}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot aggregate missing columns: {missing}")

    sorted_frame = frame.sort([id_col, time_col])
    if not window_sizes:
        features = _aggregate_full_history(
            sorted_frame, id_col, value_cols, categorical_cols
        )
    else:
        feature_views = [
            _aggregate_window(sorted_frame, id_col, value_cols, categorical_cols, window)
            for window in window_sizes
        ]
        features = feature_views[0]
        for view in feature_views[1:]:
            features = features.join(view, on=id_col, how="inner", validate="1:1")

    return features.sort(id_col).to_pandas()


def _aggregate_full_history(
    frame: pl.DataFrame,
    id_col: str,
    value_cols: list[str],
    categorical_cols: list[str],
) -> pl.DataFrame:
    return (
        frame.group_by(id_col)
        .agg(
            *[pl.col(c).mean().alias(f"{c}_mean") for c in value_cols],
            *[pl.col(c).std().fill_null(0.0).alias(f"{c}_std") for c in value_cols],
            *[pl.col(c).min().alias(f"{c}_min") for c in value_cols],
            *[pl.col(c).max().alias(f"{c}_max") for c in value_cols],
            *[pl.col(c).last().alias(f"{c}_last") for c in value_cols],
            *[
                (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_diff")
                for c in value_cols
            ],
            *[
                pl.col(c).cast(pl.String).last().alias(f"{c}_last")
                for c in categorical_cols
            ],
        )
    )


def _aggregate_window(
    frame: pl.DataFrame,
    id_col: str,
    value_cols: list[str],
    categorical_cols: list[str],
    window: int,
) -> pl.DataFrame:
    def suffix(name: str) -> str:
        return f"{name}_{window}m"

    numeric_expressions = []
    for column in value_cols:
        values = pl.col(column).tail(window)
        observations = pl.col(id_col).tail(window).count()
        difference = values.last() - values.first()
        slope = (
            pl.when(observations > 1)
            .then(difference / (observations - 1))
            .otherwise(0.0)
        )
        numeric_expressions.extend(
            [
                values.mean().alias(suffix(f"{column}_mean")),
                values.std().fill_null(0.0).alias(suffix(f"{column}_std")),
                values.min().alias(suffix(f"{column}_min")),
                values.max().alias(suffix(f"{column}_max")),
                values.first().alias(suffix(f"{column}_first")),
                values.last().alias(suffix(f"{column}_last")),
                difference.alias(suffix(f"{column}_diff")),
                slope.alias(suffix(f"{column}_slope")),
            ]
        )

    categorical_expressions = []
    for column in categorical_cols:
        states = pl.col(column).cast(pl.String).tail(window)
        transitions = (states != states.shift(1)).sum()
        categorical_expressions.extend(
            [
                states.last().alias(suffix(f"{column}_last")),
                states.n_unique().alias(suffix(f"{column}_n_states")),
                transitions.alias(suffix(f"{column}_transitions")),
            ]
        )

    return frame.group_by(id_col).agg(*numeric_expressions, *categorical_expressions)
