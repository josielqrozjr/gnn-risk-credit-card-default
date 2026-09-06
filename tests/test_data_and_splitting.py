from datetime import date

import polars as pl
import pytest

from src.data.parquet_dataset import DatasetContractError, ParquetDatasetRepository
from src.preprocessing.splitting import sample_and_split_customers


def _write_dataset(tmp_path):
    first = pl.DataFrame(
        {
            "customer_ID": ["c0", "c0", "c1"],
            "S_2": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1)],
            "P_2": [1.0, 2.0, 3.0],
            "D_63": ["A", "B", "A"],
        }
    )
    second = pl.DataFrame(
        {
            "customer_ID": ["c2", "c2", "c3"],
            "S_2": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1)],
            "P_2": [4.0, 5.0, 6.0],
            "D_63": ["B", "A", "B"],
        }
    )
    first.write_parquet(tmp_path / "data_0.parquet")
    second.write_parquet(tmp_path / "data_1.parquet")
    pl.DataFrame(
        {"customer_ID": ["c0", "c1", "c2", "c3"], "target": [0, 1, 0, 1]}
    ).write_parquet(tmp_path / "labels.parquet")


def test_repository_validates_and_profiles_partitions(tmp_path):
    _write_dataset(tmp_path)
    repository = ParquetDatasetRepository(
        str(tmp_path / "data_*.parquet"),
        str(tmp_path / "labels.parquet"),
        "customer_ID",
        "S_2",
        "target",
    )

    schema = repository.validate_features()
    labels = repository.load_labels()
    profile = repository.profile(schema, labels)

    assert schema.numeric_columns == ("P_2",)
    assert schema.categorical_columns == ("D_63",)
    assert profile["rows"] == 6
    assert profile["customers"] == 4
    assert profile["target_distribution"] == {"0": 2, "1": 2}


def test_repository_rejects_duplicate_labels(tmp_path):
    _write_dataset(tmp_path)
    pl.DataFrame(
        {"customer_ID": ["c0", "c0", "c1"], "target": [0, 0, 1]}
    ).write_parquet(tmp_path / "labels.parquet")
    repository = ParquetDatasetRepository(
        str(tmp_path / "data_*.parquet"),
        str(tmp_path / "labels.parquet"),
        "customer_ID",
        "S_2",
        "target",
    )

    with pytest.raises(DatasetContractError, match="one row per customer"):
        repository.load_labels()


def test_sampling_is_deterministic_and_disjoint():
    labels = pl.DataFrame(
        {
            "customer_ID": [f"c{i:03d}" for i in range(100)],
            "target": [i % 2 for i in range(100)],
        }
    )
    arguments = (labels, "customer_ID", "target", 50, 0.64, 0.16, 0.20, 42)

    first = sample_and_split_customers(*arguments)
    second = sample_and_split_customers(*arguments)

    assert first == second
    assert [len(first.train_ids), len(first.validation_ids), len(first.test_ids)] == [32, 8, 10]
    assert not set(first.train_ids) & set(first.validation_ids)
    assert not set(first.train_ids) & set(first.test_ids)
    assert not set(first.validation_ids) & set(first.test_ids)