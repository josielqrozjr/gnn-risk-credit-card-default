"""Inductive full-batch GNN experiments for one temporal feature view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data

from src.graph.knn_graph import (
    build_inductive_knn_edge_index,
    build_knn_edge_index,
)
from src.models.gnn import (
    GNNTrainingSummary,
    GraphClassifier,
    extract_gnn_embeddings,
    fit_gnn_classifier,
    predict_gnn_probabilities,
)
from src.preprocessing.graph_scaling import GraphFeatureScaler


@dataclass(frozen=True)
class GNNExperimentResult:
    model: GraphClassifier
    scaler: GraphFeatureScaler
    training: GNNTrainingSummary
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    validation_embeddings: np.ndarray
    test_embeddings: np.ndarray
    training_edge_count: int
    validation_edge_count: int
    test_edge_count: int


def run_inductive_gnn_experiment(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    architecture: Literal["gcn", "graphsage", "gat"] = "gat",
    k: int = 10,
    hidden_channels: int = 32,
    heads: int = 4,
    dropout: float = 0.2,
    learning_rate: float = 0.005,
    weight_decay: float = 5e-4,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    use_faiss: bool = True,
    device: str = "auto",
) -> GNNExperimentResult:
    """Train on a train-only graph and infer held-out customers as queries."""
    if validation_features.shape[0] == 0 or test_features.shape[0] == 0:
        raise ValueError("validation and test partitions must not be empty")
    if train_targets.ndim != 1 or validation_targets.ndim != 1:
        raise ValueError("target arrays must be one-dimensional")
    if train_targets.shape[0] != train_features.shape[0]:
        raise ValueError("train_targets must contain one value per training node")
    if validation_targets.shape[0] != validation_features.shape[0]:
        raise ValueError("validation_targets must contain one value per validation node")

    scaler = GraphFeatureScaler()
    train_x = scaler.fit_transform(train_features)
    validation_x = scaler.transform(validation_features)
    test_x = scaler.transform(test_features)

    training_edges = build_knn_edge_index(
        train_x, k=k, use_faiss=use_faiss, include_self=False
    )
    validation_edges = build_inductive_knn_edge_index(
        train_x, validation_x, k=k, use_faiss=use_faiss
    )
    test_edges = build_inductive_knn_edge_index(
        train_x, test_x, k=k, use_faiss=use_faiss
    )

    torch_device = _resolve_device(device)
    train_graph = _graph_data(train_x, training_edges).to(torch_device)
    validation_graph = _query_graph(
        train_x, validation_x, training_edges, validation_edges
    ).to(torch_device)
    test_graph = _query_graph(train_x, test_x, training_edges, test_edges).to(torch_device)

    torch.manual_seed(seed)
    model = GraphClassifier(
        architecture=architecture,
        in_channels=train_x.shape[1],
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
    ).to(torch_device)
    training = fit_gnn_classifier(
        model,
        train_graph,
        torch.as_tensor(train_targets, dtype=torch.float32),
        validation_graph,
        torch.as_tensor(validation_targets, dtype=torch.float32),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
    )
    validation_count = validation_x.shape[0]
    test_count = test_x.shape[0]
    return GNNExperimentResult(
        model=model,
        scaler=scaler,
        training=training,
        validation_probabilities=predict_gnn_probabilities(
            model, validation_graph, validation_count
        ),
        test_probabilities=predict_gnn_probabilities(model, test_graph, test_count),
        validation_embeddings=extract_gnn_embeddings(
            model, validation_graph, validation_count
        ),
        test_embeddings=extract_gnn_embeddings(model, test_graph, test_count),
        training_edge_count=training_edges.shape[1],
        validation_edge_count=validation_edges.shape[1],
        test_edge_count=test_edges.shape[1],
    )


def _graph_data(features: np.ndarray, edge_index: np.ndarray) -> Data:
    return Data(
        x=torch.from_numpy(np.ascontiguousarray(features)),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)),
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(device)


def _query_graph(
    train_features: np.ndarray,
    query_features: np.ndarray,
    training_edges: np.ndarray,
    query_edges: np.ndarray,
) -> Data:
    features = np.concatenate([train_features, query_features], axis=0)
    edge_index = np.concatenate([training_edges, query_edges], axis=1)
    return _graph_data(features, edge_index)
