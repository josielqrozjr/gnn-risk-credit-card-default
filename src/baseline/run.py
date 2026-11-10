"""Run the canonical tabular baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.data.parquet_dataset import DatasetContractError
from src.mvp.config import MvpConfig
from src.mvp.pipeline import MvpPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LightGBM tabular baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--sample-customers", type=int)
    parser.add_argument("--artifacts-dir")
    arguments = parser.parse_args()
    try:
        config = MvpConfig.from_file(arguments.config)
        if arguments.sample_customers is not None:
            config = replace(
                config,
                data=replace(config.data, sample_customers=arguments.sample_customers),
            )
        if arguments.artifacts_dir is not None:
            config = replace(config, artifacts_dir=arguments.artifacts_dir)
        config.validate()
        MvpPipeline(config).run()
    except (DatasetContractError, ValueError) as error:
        print(f"Baseline contract error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
