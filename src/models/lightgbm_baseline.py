"""LightGBM baseline adapter for the implementation MVP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


class LightGBMBaseline:
    """Small deterministic wrapper around LightGBM binary classification."""

    def __init__(self, seed: int, **parameters: int | float) -> None:
        self.model = LGBMClassifier(
            objective="binary",
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **parameters,
        )

    def fit(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> None:
        """Fit with validation-only early stopping."""
        self.model.fit(
            features,
            targets,
            eval_X=validation_features,
            eval_y=validation_targets,
            eval_metric="auc",
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gain": self.model.booster_.feature_importance(
                    importance_type="gain"
                ),
            }
        ).sort_values("importance_gain", ascending=False, ignore_index=True)

    def save(self, path: str | Path) -> None:
        self.model.booster_.save_model(str(path))