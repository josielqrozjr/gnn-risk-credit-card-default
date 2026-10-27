"""Training-fitted scaling for customer similarity graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class GraphFeatureScaler:
    """Standardize graph inputs using training nodes only."""

    scaler: StandardScaler = field(default_factory=StandardScaler)
    n_features_in_: int | None = None

    def fit(self, train_features: np.ndarray) -> "GraphFeatureScaler":
        _validate_matrix(train_features)
        if train_features.shape[0] == 0:
            raise ValueError("at least one training row is required")
        self.scaler.fit(train_features)
        self.n_features_in_ = train_features.shape[1]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        _validate_matrix(features)
        if self.n_features_in_ is None:
            raise RuntimeError("GraphFeatureScaler must be fitted before transform")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("feature width differs from the training feature width")
        if features.shape[0] == 0:
            return np.empty((0, self.n_features_in_), dtype=np.float32)
        return self.scaler.transform(features).astype(np.float32, copy=False)

    def fit_transform(self, train_features: np.ndarray) -> np.ndarray:
        return self.fit(train_features).transform(train_features)


def _validate_matrix(features: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if not np.isfinite(features).all():
        raise ValueError("features must only contain finite values")
