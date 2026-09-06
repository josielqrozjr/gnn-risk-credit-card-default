import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl


def test_mvp_cli_generates_sanitized_reproducible_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir()

    rows = []
    labels = []
    for customer_index in range(120):
        customer_id = f"private_customer_{customer_index:03d}"
        target = customer_index % 2
        labels.append({"customer_ID": customer_id, "target": target})
        for month in range(1, 4):
            rows.append(
                {
                    "customer_ID": customer_id,
                    "S_2": date(2020, month, 1),
                    "P_2": float(target * 5 + month + customer_index / 1000),
                    "B_1": float(customer_index % 7 + month),
                    "D_63": "risk" if target else "stable",
                }
            )

    midpoint = len(rows) // 2
    pl.DataFrame(rows[:midpoint]).write_parquet(data_dir / "data_0.parquet")
    pl.DataFrame(rows[midpoint:]).write_parquet(data_dir / "data_1.parquet")
    pl.DataFrame(labels).write_parquet(data_dir / "labels.parquet")

    config = {
        "data": {
            "features_glob": str(data_dir / "data_*.parquet"),
            "labels_path": str(data_dir / "labels.parquet"),
            "id_column": "customer_ID",
            "time_column": "S_2",
            "target_column": "target",
            "sample_customers": 100,
        },
        "split": {
            "train_fraction": 0.64,
            "validation_fraction": 0.16,
            "test_fraction": 0.20,
        },
        "model": {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": -1,
            "n_jobs": 1,
        },
        "evaluation": {"ece_bins": 5},
        "artifacts_dir": str(artifacts_dir),
        "seed": 42,
    }
    config_path = tmp_path / "mvp.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "src.mvp.run", "--config", str(config_path)],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    required = {
        "config.json",
        "data_profile.json",
        "environment.json",
        "execution.log",
        "feature_importance.csv",
        "manifest.json",
        "metrics.json",
        "model.txt",
        "preprocessor.joblib",
        "run_status.json",
        "split_summary.json",
    }
    assert required.issubset(path.name for path in artifacts_dir.iterdir())
    assert json.loads((artifacts_dir / "run_status.json").read_text())["status"] == "success"
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    assert metrics["roc_auc"] == 1.0
    assert "target" not in (artifacts_dir / "feature_importance.csv").read_text()

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    for entry in manifest["files"]:
        content = (artifacts_dir / entry["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    for path in artifacts_dir.glob("*.json"):
        assert "private_customer_" not in path.read_text(encoding="utf-8")
    for path in artifacts_dir.glob("*.csv"):
        assert "private_customer_" not in path.read_text(encoding="utf-8")