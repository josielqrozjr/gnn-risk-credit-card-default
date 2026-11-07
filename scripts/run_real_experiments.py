"""Run bounded real-data temporal and inductive GNN experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.parquet_dataset import ParquetDatasetRepository
from src.evaluation.metrics import evaluate_binary_classifier, select_f1_threshold
from src.experiments.gnn import run_inductive_gnn_experiment
from src.graph.knn_graph import build_knn_edge_index
from src.models.gnn import extract_gnn_embeddings
from src.models.hybrid import HybridLightGBMClassifier
from src.preprocessing.splitting import sample_and_split_customers
from src.preprocessing.transformations import FeaturePreprocessor
from src.reporting.artifacts import ArtifactStore
from src.temporal_features.aggregation import aggregate_temporal_features
from torch_geometric.data import Data


def main() -> None:
    args = _parse_args()
    repository = ParquetDatasetRepository(
        args.features_glob,
        args.labels_path,
        "customer_ID",
        "S_2",
        "target",
    )
    schema = repository.validate_features()
    labels = repository.load_labels()
    splits = sample_and_split_customers(
        labels,
        "customer_ID",
        "target",
        args.sample_customers,
        0.64,
        0.16,
        0.20,
        args.seed,
    )
    split_ids = splits.as_dict()
    selected_ids = sorted({customer_id for ids in split_ids.values() for customer_id in ids})
    rows = repository.load_customer_rows(selected_ids, schema)
    target_by_id = dict(zip(labels["customer_ID"].to_list(), labels["target"].to_list(), strict=True))
    results: list[dict[str, object]] = []
    started = time.perf_counter()

    for window in args.windows:
        aggregated = aggregate_temporal_features(
            rows,
            "customer_ID",
            "S_2",
            schema.numeric_columns,
            schema.categorical_columns,
            windows=(window,),
        )
        partitions = {
            name: aggregated[aggregated["customer_ID"].isin(ids)].copy()
            for name, ids in split_ids.items()
        }
        preprocessor = FeaturePreprocessor("customer_ID")
        train_x = preprocessor.fit_transform(partitions["train"]).to_numpy(dtype=np.float32)
        validation_x = preprocessor.transform(partitions["validation"]).to_numpy(dtype=np.float32)
        test_x = preprocessor.transform(partitions["test"]).to_numpy(dtype=np.float32)
        train_y = _targets(partitions["train"]["customer_ID"], target_by_id)
        validation_y = _targets(partitions["validation"]["customer_ID"], target_by_id)
        test_y = _targets(partitions["test"]["customer_ID"], target_by_id)

        for architecture in args.architectures:
            experiment = run_inductive_gnn_experiment(
                train_x,
                train_y,
                validation_x,
                validation_y,
                test_x,
                architecture=architecture,
                k=args.k,
                hidden_channels=args.hidden_channels,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=args.seed,
            )
            threshold = select_f1_threshold(
                validation_y, experiment.validation_probabilities
            )
            metrics = evaluate_binary_classifier(
                test_y,
                experiment.test_probabilities,
                threshold,
                ece_bins=10,
            )
            results.append(
                {
                    "window_months": window,
                    "architecture": architecture,
                    "feature_count": int(train_x.shape[1]),
                    "training_edges": experiment.training_edge_count,
                    "validation_edges": experiment.validation_edge_count,
                    "test_edges": experiment.test_edge_count,
                    "best_epoch": experiment.training.best_epoch,
                    "epochs_completed": experiment.training.epochs_completed,
                    "best_validation_loss": experiment.training.best_validation_loss,
                    "metrics": metrics,
                }
            )
            if window == 13 and architecture == "graphsage":
                hybrid_metrics = _evaluate_hybrid(
                    experiment,
                    train_x,
                    validation_x,
                    test_x,
                    train_y,
                    validation_y,
                    test_y,
                )
                results.append(
                    {
                        "window_months": window,
                        "architecture": "hybrid_graphsage",
                        "feature_count": int(train_x.shape[1]),
                        "metrics": hybrid_metrics,
                    }
                )
            print(
                f"{window}m {architecture}: "
                f"AMEX={metrics['amex_score']:.4f} "
                f"ROC-AUC={metrics['roc_auc']:.4f}",
                flush=True,
            )

    output = {
        "protocol": "customer-disjoint inductive GNN evaluation",
        "sample_customers": args.sample_customers,
        "split": {"train": 0.64, "validation": 0.16, "test": 0.20},
        "seed": args.seed,
        "k": args.k,
        "hidden_channels": args.hidden_channels,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "results": results,
        "total_seconds": time.perf_counter() - started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    ArtifactStore(output_path.parent).write_manifest(
        {"command": "python -m scripts.run_real_experiments", "seed": args.seed}
    )


def _evaluate_hybrid(
    experiment: object,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float | int]:
    scaled_train = experiment.scaler.transform(train_x)
    training_edges = build_knn_edge_index(scaled_train, k=10, include_self=False)
    training_graph = Data(
        x=torch.from_numpy(np.ascontiguousarray(scaled_train)),
        edge_index=torch.from_numpy(np.ascontiguousarray(training_edges)),
    )
    train_embeddings = extract_gnn_embeddings(
        experiment.model, training_graph, scaled_train.shape[0]
    )
    validation_embeddings = experiment.validation_embeddings
    test_embeddings = experiment.test_embeddings
    model = HybridLightGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=1,
        random_state=42,
    )
    model.fit(train_x, train_embeddings, train_y)
    validation_probabilities = model.predict_proba(validation_x, validation_embeddings)[:, 1]
    threshold = select_f1_threshold(validation_y, validation_probabilities)
    test_probabilities = model.predict_proba(test_x, test_embeddings)[:, 1]
    return evaluate_binary_classifier(test_y, test_probabilities, threshold, ece_bins=10)


def _targets(customer_ids: object, target_by_id: dict[str, int]) -> np.ndarray:
    return np.asarray([target_by_id[str(customer_id)] for customer_id in customer_ids], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-glob", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-customers", type=int, default=10_000)
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 6, 13])
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["gcn", "graphsage", "gat"],
        default=["gcn", "graphsage", "gat"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
