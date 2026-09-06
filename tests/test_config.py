import json

import pytest

from src.mvp.config import MvpConfig


def test_config_rejects_invalid_split(tmp_path):
    config = {
        "data": {
            "features_glob": "data_*.parquet",
            "labels_path": "labels.parquet",
            "id_column": "customer_ID",
            "time_column": "S_2",
            "target_column": "target",
            "sample_customers": 10,
        },
        "split": {
            "train_fraction": 0.5,
            "validation_fraction": 0.3,
            "test_fraction": 0.3,
        },
        "model": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": -1,
            "n_jobs": 1,
        },
        "evaluation": {"ece_bins": 5},
        "artifacts_dir": "artifacts",
        "seed": 42,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1.0"):
        MvpConfig.from_file(path)