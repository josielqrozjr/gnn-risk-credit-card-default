"""Generate a deterministic synthetic checkpoint and evidence package."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.mvp.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    MvpConfig,
    SplitConfig,
)
from src.mvp.pipeline import MvpPipeline
from src.reporting.artifacts import ArtifactStore


def main() -> None:
    root = Path("artifacts/mvp/smoke")
    random = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="gnn-risk-mvp-smoke-") as directory:
        data_dir = Path(directory)
        rows: list[dict] = []
        labels: list[dict] = []
        for customer_index in range(240):
            customer_id = f"synthetic_{customer_index:04d}"
            target = customer_index % 2
            labels.append({"customer_ID": customer_id, "target": target})
            for month in range(1, 7):
                rows.append(
                    {
                        "customer_ID": customer_id,
                        "S_2": date(2020, month, 1),
                        "P_2": float(0.8 - 0.5 * target + random.normal(0, 0.08)),
                        "B_1": float(0.2 + 0.6 * target + random.normal(0, 0.1)),
                        "S_3": float(month / 10 + random.normal(0, 0.05)),
                        "D_63": "risk" if target else "stable",
                    }
                )

        midpoint = len(rows) // 2
        pl.DataFrame(rows[:midpoint]).write_parquet(data_dir / "data_0.parquet")
        pl.DataFrame(rows[midpoint:]).write_parquet(data_dir / "data_1.parquet")
        pl.DataFrame(labels).write_parquet(data_dir / "labels.parquet")

        config = MvpConfig(
            data=DataConfig(
                features_glob=str(data_dir / "data_*.parquet"),
                labels_path=str(data_dir / "labels.parquet"),
                id_column="customer_ID",
                time_column="S_2",
                target_column="target",
                sample_customers=200,
            ),
            split=SplitConfig(0.64, 0.16, 0.20),
            model=ModelConfig(40, 0.05, 15, -1, 1),
            evaluation=EvaluationConfig(10),
            artifacts_dir=str(root),
            seed=42,
        )
        metrics = MvpPipeline(config).run()

    store = ArtifactStore(root)
    store.write_json(
        "smoke_context.json",
        {
            "dataset": "deterministic synthetic data",
            "purpose": "pipeline validation only",
            "customer_count": 200,
            "report_as_amex_baseline": False,
        },
    )
    store.write_manifest(
        {
            "command": "python -m scripts.generate_smoke_artifacts",
            "seed": 42,
        }
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()