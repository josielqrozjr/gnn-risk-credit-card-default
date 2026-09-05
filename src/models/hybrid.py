"""Hybrid model utilities combining tabular features and GNN embeddings."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier


class HybridLightGBMClassifier:
    """Train LightGBM on concatenated tabular + graph embedding features."""

    def __init__(self, **lgbm_params) -> None:
        params = {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42}
        params.update(lgbm_params)
        self.model = LGBMClassifier(**params)

    def fit(self, tabular_features: np.ndarray, gnn_embeddings: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier on concatenated tabular and embedding features."""
        x = self._combine_features(tabular_features, gnn_embeddings)
        self.model.fit(x, y)

    def predict_proba(self, tabular_features: np.ndarray, gnn_embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities from concatenated feature space."""
        x = self._combine_features(tabular_features, gnn_embeddings)
        return self.model.predict_proba(x)

    def predict(self, tabular_features: np.ndarray, gnn_embeddings: np.ndarray) -> np.ndarray:
        """Predict class labels from concatenated feature space."""
        x = self._combine_features(tabular_features, gnn_embeddings)
        return self.model.predict(x)

    @staticmethod
    def _combine_features(tabular_features: np.ndarray, gnn_embeddings: np.ndarray) -> np.ndarray:
        return np.concatenate([tabular_features, gnn_embeddings], axis=1)
