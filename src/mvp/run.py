"""Compatibility entry point for the tabular baseline."""

from __future__ import annotations

import argparse

from src.data.parquet_dataset import DatasetContractError
from src.mvp.config import MvpConfig
from src.mvp.pipeline import MvpPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LightGBM tabular baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    arguments = parser.parse_args()

    try:
        config = MvpConfig.from_file(arguments.config)
        MvpPipeline(config).run()
    except (DatasetContractError, ValueError) as error:
        print(f"Baseline contract error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())