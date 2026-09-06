"""Deterministic customer sampling and disjoint data splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class CustomerSplits:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train_ids,
            "validation": self.validation_ids,
            "test": self.test_ids,
        }


def sample_and_split_customers(
    labels: pl.DataFrame,
    id_column: str,
    target_column: str,
    sample_customers: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> CustomerSplits:
    """Return deterministic target-stratified and customer-disjoint splits."""
    ids = labels[id_column].to_numpy()
    targets = labels[target_column].to_numpy()

    if sample_customers < len(ids):
        ids, _, targets, _ = train_test_split(
            ids,
            targets,
            train_size=sample_customers,
            stratify=targets,
            random_state=seed,
        )

    development_ids, test_ids, development_targets, _ = train_test_split(
        ids,
        targets,
        test_size=test_fraction,
        stratify=targets,
        random_state=seed,
    )
    relative_validation_fraction = validation_fraction / (train_fraction + validation_fraction)
    train_ids, validation_ids = train_test_split(
        development_ids,
        test_size=relative_validation_fraction,
        stratify=development_targets,
        random_state=seed,
    )

    splits = CustomerSplits(
        train_ids=tuple(sorted(np.asarray(train_ids, dtype=str).tolist())),
        validation_ids=tuple(sorted(np.asarray(validation_ids, dtype=str).tolist())),
        test_ids=tuple(sorted(np.asarray(test_ids, dtype=str).tolist())),
    )
    _assert_disjoint(splits)
    return splits


def summarize_splits(
    splits: CustomerSplits,
    labels: pl.DataFrame,
    id_column: str,
    target_column: str,
) -> dict[str, dict[str, int]]:
    """Summarize split sizes and class counts without exposing identifiers."""
    summary: dict[str, dict[str, int]] = {}
    for name, customer_ids in splits.as_dict().items():
        subset = labels.filter(pl.col(id_column).is_in(customer_ids))
        counts = subset.group_by(target_column).len()
        summary[name] = {
            "customers": subset.height,
            "target_0": _target_count(counts, target_column, 0),
            "target_1": _target_count(counts, target_column, 1),
        }
    return summary


def _target_count(counts: pl.DataFrame, target_column: str, value: int) -> int:
    row = counts.filter(pl.col(target_column) == value)
    return 0 if row.is_empty() else int(row["len"][0])


def _assert_disjoint(splits: CustomerSplits) -> None:
    train = set(splits.train_ids)
    validation = set(splits.validation_ids)
    test = set(splits.test_ids)
    if train & validation or train & test or validation & test:
        raise RuntimeError("Customer splits must be disjoint")