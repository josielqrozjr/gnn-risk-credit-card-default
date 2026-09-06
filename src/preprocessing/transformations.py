"""Train-fitted transformations for aggregated customer features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class FeaturePreprocessor:
    """Impute and encode features without fitting on held-out customers."""

    id_column: str
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str] = field(default_factory=dict)
    categorical_mappings: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        """Learn usable columns, imputations, and encodings from training data."""
        candidates = [column for column in frame.columns if column != self.id_column]
        usable = [
            column
            for column in candidates
            if frame[column].notna().any() and frame[column].nunique(dropna=True) > 1
        ]
        self.numeric_columns = [
            column for column in usable if pd.api.types.is_numeric_dtype(frame[column])
        ]
        self.categorical_columns = [
            column for column in usable if column not in self.numeric_columns
        ]

        numeric = frame[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        self.numeric_medians = {
            column: float(numeric[column].median()) for column in self.numeric_columns
        }
        for column in self.categorical_columns:
            values = frame[column].dropna().astype(str)
            mode = values.mode().iloc[0]
            categories = sorted(values.unique().tolist())
            self.categorical_modes[column] = mode
            self.categorical_mappings[column] = {
                category: index for index, category in enumerate(categories)
            }
        if not self.feature_names:
            raise ValueError("No usable training features remain after preprocessing")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the training feature contract to another customer partition."""
        missing = sorted(set(self.feature_names).difference(frame.columns))
        if missing:
            raise ValueError(f"Input is missing fitted feature columns: {missing}")

        transformed: dict[str, pd.Series] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            transformed[column] = values.fillna(self.numeric_medians[column]).astype(
                "float32"
            )
        for column in self.categorical_columns:
            values = frame[column].fillna(self.categorical_modes[column]).astype(str)
            transformed[column] = (
                values.map(self.categorical_mappings[column]).fillna(-1).astype("float32")
            )
        return pd.DataFrame(transformed, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric_columns, *self.categorical_columns]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        return joblib.load(path)