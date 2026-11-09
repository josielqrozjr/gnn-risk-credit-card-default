"""Evaluate a leakage-safe OOF GraphSAGE hybrid on real AMEX data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from src.data.parquet_dataset import ParquetDatasetRepository
from src.evaluation.metrics import evaluate_binary_classifier, select_f1_threshold
from src.experiments.gnn import run_inductive_gnn_experiment
from src.models.hybrid import HybridLightGBMClassifier
from src.preprocessing.splitting import sample_and_split_customers
from src.preprocessing.transformations import FeaturePreprocessor
from src.temporal_features.aggregation import aggregate_temporal_features


def main() -> None:
    args = _parse_args()
    repository = ParquetDatasetRepository(
        args.features_glob, args.labels_path, "customer_ID", "S_2", "target"
    )
    schema = repository.validate_features()
    labels = repository.load_labels()
    splits = sample_and_split_customers(
        labels, "customer_ID", "target", args.sample_customers,
        0.64, 0.16, 0.20, args.seed,
    )
    ids = splits.as_dict()
    selected_ids = sorted({customer_id for values in ids.values() for customer_id in values})
    rows = repository.load_customer_rows(selected_ids, schema)
    target_by_id = dict(zip(labels["customer_ID"].to_list(), labels["target"].to_list(), strict=True))
    aggregated = aggregate_temporal_features(
        rows, "customer_ID", "S_2", schema.numeric_columns,
        schema.categorical_columns, windows=(args.window,),
    )
    partitions = {
        name: aggregated[aggregated["customer_ID"].isin(customer_ids)].copy()
        for name, customer_ids in ids.items()
    }

    tabular_preprocessor = FeaturePreprocessor("customer_ID")
    tabular = {
        name: tabular_preprocessor.fit_transform(partitions["train"]).to_numpy(dtype=np.float32)
        if name == "train"
        else tabular_preprocessor.transform(partitions[name]).to_numpy(dtype=np.float32)
        for name in ids
    }
    targets = {
        name: np.asarray(
            [target_by_id[str(customer_id)] for customer_id in partitions[name]["customer_ID"]],
            dtype=np.float32,
        )
        for name in ids
    }

    oof_embeddings = _build_oof_embeddings(
        tabular["train"], targets["train"], args, args.seed
    )
    final_experiment = run_inductive_gnn_experiment(
        tabular["train"], targets["train"], tabular["validation"], targets["validation"],
        tabular["test"], architecture="graphsage", k=args.k,
        hidden_channels=args.hidden_channels, max_epochs=args.max_epochs,
        patience=args.patience, seed=args.seed, device=args.device,
        use_faiss=args.use_faiss,
    )
    hybrid = HybridLightGBMClassifier(
        n_estimators=args.n_estimators, learning_rate=0.05,
        num_leaves=31, n_jobs=1, random_state=args.seed,
    )
    hybrid.fit(tabular["train"], oof_embeddings, targets["train"])
    validation_probabilities = hybrid.predict_proba(
        tabular["validation"], final_experiment.validation_embeddings
    )[:, 1]
    threshold = select_f1_threshold(targets["validation"], validation_probabilities)
    test_probabilities = hybrid.predict_proba(
        tabular["test"], final_experiment.test_embeddings
    )[:, 1]
    metrics = evaluate_binary_classifier(
        targets["test"], test_probabilities, threshold, ece_bins=10
    )
    output = {
        "protocol": "three-fold out-of-fold GraphSAGE hybrid",
        "window_months": args.window,
        "sample_customers": args.sample_customers,
        "folds": args.folds,
        "seed": args.seed,
        "k": args.k,
        "hidden_channels": args.hidden_channels,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "device": str(next(final_experiment.model.parameters()).device),
        "metrics": metrics,
        "total_seconds": time.perf_counter() - args.started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


def _build_oof_embeddings(
    features: np.ndarray, targets: np.ndarray, args: argparse.Namespace, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    rng.shuffle(positive)
    rng.shuffle(negative)
    folds = [[] for _ in range(args.folds)]
    for values in (positive, negative):
        for index, value in enumerate(values):
            folds[index % args.folds].append(int(value))
    embeddings = np.empty((features.shape[0], args.hidden_channels), dtype=np.float32)
    for fold, held_out in enumerate(folds):
        held_out_indices = np.asarray(sorted(held_out), dtype=np.int64)
        train_indices = np.setdiff1d(np.arange(features.shape[0]), held_out_indices)
        result = run_inductive_gnn_experiment(
            features[train_indices], targets[train_indices],
            features[held_out_indices], targets[held_out_indices],
            features[held_out_indices],
            architecture="graphsage", k=args.k,
            hidden_channels=args.hidden_channels, max_epochs=args.max_epochs,
            patience=args.patience, seed=seed + fold + 1, device=args.device,
            use_faiss=args.use_faiss,
        )
        embeddings[held_out_indices] = result.validation_embeddings
        print(f"OOF fold {fold + 1}/{args.folds}: {len(held_out_indices)} customers", flush=True)
    return embeddings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-glob", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-customers", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=13)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-faiss",
        dest="use_faiss",
        action="store_false",
        default=True,
        help="Use scikit-learn instead of FAISS for CPU kNN graph construction.",
    )
    args = parser.parse_args()
    args.started = time.perf_counter()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    return args


if __name__ == "__main__":
    main()
