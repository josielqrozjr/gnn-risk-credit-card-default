"""Configuration loading and validation for the MVP pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    features_glob: str
    labels_path: str
    id_column: str
    time_column: str
    target_column: str
    sample_customers: int


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    n_jobs: int


@dataclass(frozen=True)
class EvaluationConfig:
    ece_bins: int


@dataclass(frozen=True)
class MvpConfig:
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    artifacts_dir: str
    seed: int

    @classmethod
    def from_file(cls, path: str | Path) -> "MvpConfig":
        """Load and validate a JSON configuration file."""
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            raw = json.load(config_file)

        try:
            config = cls(
                data=DataConfig(**raw["data"]),
                split=SplitConfig(**raw["split"]),
                model=ModelConfig(**raw["model"]),
                evaluation=EvaluationConfig(**raw["evaluation"]),
                artifacts_dir=raw["artifacts_dir"],
                seed=raw["seed"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid MVP configuration: {error}") from error

        config.validate()
        return config

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent settings."""
        split_total = (
            self.split.train_fraction
            + self.split.validation_fraction
            + self.split.test_fraction
        )
        if abs(split_total - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0")
        if min(
            self.split.train_fraction,
            self.split.validation_fraction,
            self.split.test_fraction,
        ) <= 0:
            raise ValueError("Every split fraction must be positive")
        if self.data.sample_customers < 1:
            raise ValueError("sample_customers must be at least 1")
        if self.model.n_estimators < 1 or self.model.num_leaves < 2:
            raise ValueError("LightGBM capacity settings must be positive")
        if self.model.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if self.evaluation.ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return {
            "data": vars(self.data),
            "split": vars(self.split),
            "model": vars(self.model),
            "evaluation": vars(self.evaluation),
            "artifacts_dir": self.artifacts_dir,
            "seed": self.seed,
        }