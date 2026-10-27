"""Validated access to partitioned AMEX feature and label data."""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class DatasetContractError(ValueError):
    """Raised when local data does not satisfy the MVP contract."""


@dataclass(frozen=True)
class DatasetSchema:
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


class ParquetDatasetRepository:
    """Repository for partitioned features and customer-level labels."""

    AMEX_CATEGORICAL_COLUMNS = frozenset(
        {
            "B_30",
            "B_38",
            "D_63",
            "D_64",
            "D_66",
            "D_68",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
        }
    )

    def __init__(
        self,
        features_glob: str,
        labels_path: str,
        id_column: str,
        time_column: str,
        target_column: str,
    ) -> None:
        self.features_glob = features_glob
        self.labels_path = Path(labels_path)
        self.id_column = id_column
        self.time_column = time_column
        self.target_column = target_column
        self.feature_paths = tuple(Path(path) for path in sorted(glob(features_glob)))

    def validate_features(self) -> DatasetSchema:
        """Validate partition schemas and return the model feature contract."""
        if not self.feature_paths:
            raise DatasetContractError(
                f"No feature partitions matched {self.features_glob!r}"
            )

        schemas = [pq.ParquetFile(path).schema_arrow for path in self.feature_paths]
        reference = schemas[0]
        for path, schema in zip(self.feature_paths[1:], schemas[1:], strict=True):
            if not schema.equals(reference):
                raise DatasetContractError(
                    f"Feature partition {path} has a different Parquet schema"
                )

        required = {self.id_column, self.time_column}
        missing = sorted(required.difference(reference.names))
        if missing:
            raise DatasetContractError(
                f"Feature partitions are missing required columns: {missing}"
            )

        time_type = reference.field(self.time_column).type
        if not (pa.types.is_date(time_type) or pa.types.is_timestamp(time_type)):
            raise DatasetContractError(
                f"{self.time_column} must be a date or timestamp, found {time_type}"
            )

        excluded = {self.id_column, self.time_column, self.target_column}
        feature_columns = tuple(name for name in reference.names if name not in excluded)
        categorical = tuple(
            name for name in feature_columns if name in self.AMEX_CATEGORICAL_COLUMNS
        )
        numeric = tuple(name for name in feature_columns if name not in categorical)
        if not numeric:
            raise DatasetContractError("No numeric model features were found")
        return DatasetSchema(feature_columns, categorical, numeric)

    def load_labels(self) -> pl.DataFrame:
        """Load and validate one binary target per customer."""
        if not self.labels_path.exists():
            raise DatasetContractError(
                "Customer labels are required but were not found at "
                f"{self.labels_path}. Add the AMEX train_labels file or Parquet directory."
            )

        if self.labels_path.is_dir():
            label_paths = tuple(
                sorted(
                    (*self.labels_path.glob("*.parquet"), *self.labels_path.glob("*.pq"))
                )
            )
            if not label_paths:
                raise DatasetContractError(
                    f"No Parquet label partitions found in {self.labels_path}"
                )
            labels = pl.read_parquet([str(path) for path in label_paths])
        elif self.labels_path.suffix.lower() == ".csv":
            labels = pl.read_csv(self.labels_path)
        elif self.labels_path.suffix.lower() in {".parquet", ".pq"}:
            labels = pl.read_parquet(self.labels_path)
        else:
            raise DatasetContractError(
                "Labels must be stored as CSV, Parquet, or a directory of Parquet partitions"
            )
        required = {self.id_column, self.target_column}
        missing = sorted(required.difference(labels.columns))
        if missing:
            raise DatasetContractError(f"Labels are missing required columns: {missing}")

        labels = labels.select(self.id_column, self.target_column)
        if labels[self.id_column].null_count() or labels[self.target_column].null_count():
            raise DatasetContractError("Labels must not contain null identifiers or targets")
        if labels[self.id_column].n_unique() != labels.height:
            raise DatasetContractError("Labels must contain exactly one row per customer")

        target_values = set(labels[self.target_column].unique().to_list())
        if not target_values.issubset({0, 1}) or len(target_values) != 2:
            raise DatasetContractError(
                f"Target must contain both binary values 0 and 1, found {target_values}"
            )
        return labels.sort(self.id_column)

    def profile(self, schema: DatasetSchema, labels: pl.DataFrame | None = None) -> dict:
        """Build a non-sensitive profile from metadata and narrow scans."""
        metadata = [pq.ParquetFile(path).metadata for path in self.feature_paths]
        identity_profile = (
            self.scan_features()
            .select(
                pl.len().alias("rows"),
                pl.col(self.id_column).n_unique().alias("customers"),
                pl.col(self.time_column).min().alias("date_min"),
                pl.col(self.time_column).max().alias("date_max"),
                pl.col(self.id_column).null_count().alias("null_customer_ids"),
                pl.col(self.time_column).null_count().alias("null_dates"),
                pl.struct(self.id_column, self.time_column)
                .is_duplicated()
                .sum()
                .alias("duplicate_customer_month_rows"),
            )
            .collect()
            .row(0, named=True)
        )
        if identity_profile["null_customer_ids"] or identity_profile["null_dates"]:
            raise DatasetContractError("Feature identity and date columns must not be null")
        if identity_profile["duplicate_customer_month_rows"]:
            raise DatasetContractError("Feature partitions contain duplicate customer-month rows")
        profile = {
            "feature_files": len(self.feature_paths),
            "feature_bytes": sum(path.stat().st_size for path in self.feature_paths),
            "rows": sum(item.num_rows for item in metadata),
            "row_groups": sum(item.num_row_groups for item in metadata),
            "customers": identity_profile["customers"],
            "date_min": identity_profile["date_min"].isoformat(),
            "date_max": identity_profile["date_max"].isoformat(),
            "duplicate_customer_month_rows": identity_profile[
                "duplicate_customer_month_rows"
            ],
            "raw_feature_columns": len(schema.feature_columns),
            "numeric_columns": len(schema.numeric_columns),
            "categorical_columns": len(schema.categorical_columns),
            "labels_available": labels is not None,
        }
        if labels is not None:
            counts = labels.group_by(self.target_column).len().sort(self.target_column)
            profile["label_customers"] = labels.height
            profile["target_distribution"] = {
                str(row[self.target_column]): row["len"]
                for row in counts.iter_rows(named=True)
            }
        return profile

    def scan_features(self) -> pl.LazyFrame:
        """Return a lazy scan over all feature partitions."""
        return pl.scan_parquet([str(path) for path in self.feature_paths])

    def load_customer_rows(
        self, customer_ids: list[str], schema: DatasetSchema
    ) -> pl.DataFrame:
        """Materialize only selected customers and contracted feature columns."""
        selected_columns = [
            self.id_column,
            self.time_column,
            *schema.feature_columns,
        ]
        return (
            self.scan_features()
            .filter(pl.col(self.id_column).is_in(customer_ids))
            .select(selected_columns)
            .collect()
        )