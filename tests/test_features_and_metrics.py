import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.evaluation.metrics import (
    amex_score,
    evaluate_binary_classifier,
    expected_calibration_error,
    select_f1_threshold,
)
from src.preprocessing.transformations import FeaturePreprocessor
from src.temporal_features.aggregation import aggregate_temporal_features


def test_temporal_aggregation_is_sorted_and_complete():
    raw = pl.DataFrame(
        {
            "customer_ID": ["b", "a", "a", "b"],
            "S_2": [2, 2, 1, 1],
            "value": [5.0, 3.0, 1.0, 2.0],
            "category": ["new", "late", "ok", "old"],
        }
    )

    result = aggregate_temporal_features(
        raw, "customer_ID", "S_2", ["value"], ["category"]
    )

    assert result["customer_ID"].tolist() == ["a", "b"]
    assert result["value_diff"].tolist() == [2.0, 3.0]
    assert result["category_last"].tolist() == ["late", "new"]
    assert set(result.columns) == {
        "customer_ID",
        "value_mean",
        "value_std",
        "value_min",
        "value_max",
        "value_last",
        "value_diff",
        "category_last",
    }


def test_temporal_aggregation_builds_trailing_window_views():
    raw = pl.DataFrame(
        {
            "customer_ID": ["a"] * 4,
            "S_2": [1, 2, 3, 4],
            "value": [1.0, 3.0, 7.0, 9.0],
            "category": ["x", "x", "y", "y"],
        }
    )

    result = aggregate_temporal_features(
        raw,
        "customer_ID",
        "S_2",
        ["value"],
        ["category"],
        windows=(2, 4),
    )
    row = result.iloc[0]

    assert row["value_mean_2m"] == pytest.approx(8.0)
    assert row["value_diff_2m"] == pytest.approx(2.0)
    assert row["value_slope_2m"] == pytest.approx(2.0)
    assert row["category_n_states_2m"] == 1
    assert row["category_transitions_2m"] == 0
    assert row["value_mean_4m"] == pytest.approx(5.0)
    assert row["value_slope_4m"] == pytest.approx(8 / 3)
    assert row["category_n_states_4m"] == 2
    assert row["category_transitions_4m"] == 1


@pytest.mark.parametrize("windows", [(0,), (-1,), (2, 2), (1.5,)])
def test_temporal_aggregation_rejects_invalid_windows(windows):
    raw = pl.DataFrame(
        {"customer_ID": ["a"], "S_2": [1], "value": [1.0]}
    )

    with pytest.raises(ValueError, match="windows"):
        aggregate_temporal_features(raw, "customer_ID", "S_2", ["value"], windows=windows)


def test_preprocessor_uses_training_contract_and_handles_unknowns(tmp_path):
    train = pd.DataFrame(
        {
            "customer_ID": ["a", "b", "c"],
            "numeric": [1.0, np.nan, 3.0],
            "category": ["x", "y", "x"],
            "constant": [1, 1, 1],
        }
    )
    held_out = pd.DataFrame(
        {"customer_ID": ["d"], "numeric": [np.nan], "category": ["z"], "constant": [2]}
    )
    preprocessor = FeaturePreprocessor("customer_ID").fit(train)
    transformed = preprocessor.transform(held_out)
    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path)

    assert preprocessor.feature_names == ["numeric", "category"]
    assert transformed.iloc[0].tolist() == [2.0, -1.0]
    assert FeaturePreprocessor.load(path).feature_names == preprocessor.feature_names


def test_metrics_reward_perfect_ranking():
    targets = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.01, 0.99, 0.1, 0.9, 0.2, 0.8])
    threshold = select_f1_threshold(targets, probabilities)
    metrics = evaluate_binary_classifier(targets, probabilities, threshold, 5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert amex_score(targets, probabilities) == pytest.approx(
        amex_score(targets, targets.astype(float))
    )
    assert amex_score(targets, probabilities) > amex_score(targets, 1 - probabilities)
    assert expected_calibration_error(targets, targets.astype(float), 5) == 0.0