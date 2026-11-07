"""End-to-end orchestration for the LightGBM implementation baseline."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import psutil

from src.data.parquet_dataset import ParquetDatasetRepository
from src.evaluation.metrics import evaluate_binary_classifier, select_f1_threshold
from src.models.lightgbm_baseline import LightGBMBaseline
from src.mvp.config import MvpConfig
from src.preprocessing.splitting import sample_and_split_customers, summarize_splits
from src.preprocessing.transformations import FeaturePreprocessor
from src.reporting.artifacts import ArtifactStore
from src.temporal_features.aggregation import aggregate_temporal_features


class MvpPipeline:
    """Facade that owns one reproducible baseline execution."""

    def __init__(self, config: MvpConfig) -> None:
        self.config = config
        self.artifacts = ArtifactStore(config.artifacts_dir)
        self.logger = _configure_logger(self.artifacts.path("execution.log"))
        self.stage_seconds: dict[str, float] = {}
        self.peak_rss_bytes = psutil.Process().memory_info().rss

    def run(self) -> dict[str, float | int]:
        """Execute the baseline and return held-out test metrics."""
        started = time.perf_counter()
        np.random.seed(self.config.seed)
        self.artifacts.write_json("config.json", self.config.to_dict())
        self.logger.info("Starting the LightGBM MVP pipeline")

        repository = ParquetDatasetRepository(
            features_glob=self.config.data.features_glob,
            labels_path=self.config.data.labels_path,
            id_column=self.config.data.id_column,
            time_column=self.config.data.time_column,
            target_column=self.config.data.target_column,
        )
        try:
            with self._stage("validate_and_profile_data"):
                schema = repository.validate_features()
                profile = repository.profile(schema)
                self.artifacts.write_json("data_profile.json", profile)
                labels = repository.load_labels()
                profile = repository.profile(schema, labels)
                self.artifacts.write_json("data_profile.json", profile)

            with self._stage("sample_and_split_customers"):
                splits = sample_and_split_customers(
                    labels=labels,
                    id_column=self.config.data.id_column,
                    target_column=self.config.data.target_column,
                    sample_customers=self.config.data.sample_customers,
                    train_fraction=self.config.split.train_fraction,
                    validation_fraction=self.config.split.validation_fraction,
                    test_fraction=self.config.split.test_fraction,
                    seed=self.config.seed,
                )
                split_summary = summarize_splits(
                    splits,
                    labels,
                    self.config.data.id_column,
                    self.config.data.target_column,
                )
                self.artifacts.write_json("split_summary.json", split_summary)

            with self._stage("aggregate_temporal_features"):
                selected_ids = sorted(
                    customer_id
                    for customer_ids in splits.as_dict().values()
                    for customer_id in customer_ids
                )
                customer_rows = repository.load_customer_rows(selected_ids, schema)
                aggregated = aggregate_temporal_features(
                    customer_rows,
                    id_col=self.config.data.id_column,
                    time_col=self.config.data.time_column,
                    value_cols=schema.numeric_columns,
                    categorical_cols=schema.categorical_columns,
                )
                label_frame = labels.to_pandas()
                customers = aggregated.merge(
                    label_frame,
                    on=self.config.data.id_column,
                    how="inner",
                    validate="one_to_one",
                )
                if len(customers) != len(selected_ids):
                    raise ValueError(
                        "Some sampled customers are missing from the feature partitions"
                    )

            with self._stage("preprocess_features"):
                partitions = {
                    name: customers[
                        customers[self.config.data.id_column].isin(customer_ids)
                    ].copy()
                    for name, customer_ids in splits.as_dict().items()
                }
                preprocessor = FeaturePreprocessor(self.config.data.id_column)
                feature_partitions = {
                    name: partition.drop(columns=[self.config.data.target_column])
                    for name, partition in partitions.items()
                }
                train_x = preprocessor.fit_transform(feature_partitions["train"])
                validation_x = preprocessor.transform(feature_partitions["validation"])
                test_x = preprocessor.transform(feature_partitions["test"])
                train_y = _targets(partitions["train"], self.config.data.target_column)
                validation_y = _targets(
                    partitions["validation"], self.config.data.target_column
                )
                test_y = _targets(partitions["test"], self.config.data.target_column)

            with self._stage("train_lightgbm"):
                model = LightGBMBaseline(
                    seed=self.config.seed, **vars(self.config.model)
                )
                model.fit(train_x, train_y, validation_x, validation_y)
                validation_probabilities = model.predict_proba(validation_x)
                threshold = select_f1_threshold(validation_y, validation_probabilities)

            with self._stage("evaluate_and_save"):
                test_probabilities = model.predict_proba(test_x)
                metrics = evaluate_binary_classifier(
                    test_y,
                    test_probabilities,
                    threshold,
                    self.config.evaluation.ece_bins,
                )
                self.artifacts.write_json("metrics.json", metrics)
                self.artifacts.write_csv(
                    "feature_importance.csv",
                    model.feature_importance(preprocessor.feature_names),
                )
                model.save(self.artifacts.path("model.txt"))
                preprocessor.save(self.artifacts.path("preprocessor.joblib"))

            self.artifacts.write_json(
                "run_status.json",
                {
                    "status": "success",
                    "total_seconds": time.perf_counter() - started,
                    "stage_seconds": self.stage_seconds,
                    "peak_rss_bytes": self.peak_rss_bytes,
                },
            )
            self.logger.info("MVP pipeline completed successfully")
            return metrics
        except Exception as error:
            self.logger.exception("MVP pipeline failed: %s", error)
            self.artifacts.write_json(
                "run_status.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "total_seconds": time.perf_counter() - started,
                    "stage_seconds": self.stage_seconds,
                    "peak_rss_bytes": self.peak_rss_bytes,
                },
            )
            raise
        finally:
            self.artifacts.write_json("environment.json", _environment_report())
            for handler in self.logger.handlers:
                handler.flush()
            self.artifacts.write_manifest(
                {
                    "command": "python -m src.baseline.run --config configs/baseline.json",
                    "seed": self.config.seed,
                }
            )

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        self.logger.info("Stage started: %s", name)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stage_seconds[name] = elapsed
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                psutil.Process().memory_info().rss,
                _maximum_resident_set_bytes(),
            )
            self.logger.info("Stage finished: %s in %.3f seconds", name, elapsed)


def _targets(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    return frame[target_column].to_numpy(dtype=np.int8)


def _configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("mvp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _maximum_resident_set_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _environment_report() -> dict:
    packages = [
        "joblib",
        "lightgbm",
        "numpy",
        "pandas",
        "polars",
        "psutil",
        "pyarrow",
        "scikit-learn",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": psutil.disk_usage(".").free,
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _cpu_model() -> str:
    if sys.platform != "darwin":
        return platform.processor() or platform.machine()
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or platform.processor() or platform.machine()