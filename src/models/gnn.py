"""Full-batch GNN classifiers and validation-based training utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

GNNArchitecture = Literal["gcn", "graphsage", "gat"]


class GraphClassifier(nn.Module):
    """Two-layer GCN, GraphSAGE, or GAT node classifier."""

    def __init__(
        self,
        architecture: GNNArchitecture,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if architecture not in {"gcn", "graphsage", "gat"}:
            raise ValueError(f"Unsupported GNN architecture: {architecture}")
        if min(in_channels, hidden_channels, out_channels, heads) < 1:
            raise ValueError("GNN channel counts and heads must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.architecture = architecture
        self.dropout = dropout
        if architecture == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif architecture == "graphsage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            self.conv1 = GATConv(
                in_channels, hidden_channels, heads=heads, dropout=dropout
            )
            self.conv2 = GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )

    def encode(self, data: Data) -> torch.Tensor:
        """Return hidden node representations before the output layer."""
        features = F.dropout(data.x, p=self.dropout, training=self.training)
        embeddings = self.conv1(features, data.edge_index)
        return F.relu(embeddings)

    def forward(self, data: Data) -> torch.Tensor:
        embeddings = self.encode(data)
        hidden = F.dropout(embeddings, p=self.dropout, training=self.training)
        return self.conv2(hidden, data.edge_index)


@dataclass(frozen=True)
class GNNTrainingSummary:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float


def fit_gnn_classifier(
    model: GraphClassifier,
    train_data: Data,
    train_targets: torch.Tensor,
    validation_data: Data,
    validation_targets: torch.Tensor,
    *,
    learning_rate: float = 0.005,
    weight_decay: float = 5e-4,
    max_epochs: int = 100,
    patience: int = 10,
) -> GNNTrainingSummary:
    """Train on the training graph and early-stop on directed query nodes."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if train_data.x.ndim != 2 or validation_data.x.ndim != 2:
        raise ValueError("graph features must be two-dimensional")
    if train_data.x.shape[0] == 0:
        raise ValueError("at least one training node is required")
    if train_data.x.shape[1] != validation_data.x.shape[1]:
        raise ValueError("training and validation feature widths must match")
    if validation_data.x.shape[0] <= train_data.x.shape[0]:
        raise ValueError("validation graph must append at least one query node")
    if not torch.isfinite(train_data.x).all() or not torch.isfinite(
        validation_data.x
    ).all():
        raise ValueError("graph features must be finite")
    if train_data.x.device != validation_data.x.device:
        raise ValueError("training and validation graph features must use the same device")
    if not torch.equal(validation_data.x[: train_data.x.shape[0]], train_data.x):
        raise ValueError("validation graph must preserve the training node features")

    train_targets = train_targets.reshape(-1).to(
        device=train_data.x.device, dtype=torch.float32
    )
    validation_targets = validation_targets.reshape(-1).to(
        device=validation_data.x.device, dtype=torch.float32
    )
    train_count = train_data.x.shape[0]
    query_count = validation_data.x.shape[0] - train_count
    if train_targets.numel() != train_count:
        raise ValueError("train_targets must contain one value per training node")
    if validation_targets.numel() != query_count:
        raise ValueError("validation_targets must contain one value per query node")
    for name, graph, node_count in (
        ("training", train_data, train_count),
        ("validation", validation_data, validation_data.x.shape[0]),
    ):
        if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
            raise ValueError(f"{name} edge_index must have shape (2, num_edges)")
        if graph.edge_index.numel() and (
            int(graph.edge_index.min()) < 0
            or int(graph.edge_index.max()) >= node_count
        ):
            raise ValueError(f"{name} edge_index contains an out-of-range node")
        if graph.edge_index.device != graph.x.device:
            raise ValueError(f"{name} edge_index and features must use the same device")
    if validation_data.edge_index.numel() and torch.any(
        validation_data.edge_index[0] >= train_count
    ):
        raise ValueError("validation query nodes must not send messages to the graph")
    validation_train_edges = validation_data.edge_index[
        :, validation_data.edge_index[1] < train_count
    ]
    if not torch.equal(
        _canonical_edges(train_data.edge_index, train_count),
        _canonical_edges(validation_train_edges, train_count),
    ):
        raise ValueError("validation graph must preserve all training graph edges")
    if not torch.isfinite(train_targets).all() or not torch.isfinite(
        validation_targets
    ).all():
        raise ValueError("targets must be finite")
    if not torch.all((train_targets == 0) | (train_targets == 1)) or not torch.all(
        (validation_targets == 0) | (validation_targets == 1)
    ):
        raise ValueError("targets must be binary")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_completed = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(train_data).reshape(-1)
        if train_logits.numel() != train_count:
            raise ValueError("model must return one logit per training node")
        train_loss = loss_function(train_logits, train_targets)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_data).reshape(-1)
            if validation_logits.numel() != validation_data.x.shape[0]:
                raise ValueError("model must return one logit per validation graph node")
            validation_loss = loss_function(
                validation_logits[-query_count:], validation_targets
            )
        current_validation_loss = float(validation_loss.item())
        if not np.isfinite(current_validation_loss):
            raise FloatingPointError(
                "validation loss became non-finite during GNN training"
            )

        epochs_completed = epoch
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("GNN training did not produce a valid validation checkpoint")
    model.load_state_dict(best_state)
    return GNNTrainingSummary(
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        best_validation_loss=best_validation_loss,
    )


def predict_gnn_probabilities(
    model: GraphClassifier,
    data: Data,
    query_count: int,
) -> np.ndarray:
    """Predict probabilities for the final query nodes of an augmented graph."""
    if query_count < 1 or query_count > data.x.shape[0]:
        raise ValueError("query_count must select at least one node in the graph")
    model.eval()
    with torch.no_grad():
        logits = model(data).reshape(-1)
        if logits.numel() != data.x.shape[0]:
            raise ValueError("model must return one logit per graph node")
        probabilities = torch.sigmoid(logits[-query_count:])
    return probabilities.cpu().numpy()


def extract_gnn_embeddings(
    model: GraphClassifier,
    data: Data,
    query_count: int,
) -> np.ndarray:
    """Extract frozen hidden representations for the final query nodes."""
    if query_count < 1 or query_count > data.x.shape[0]:
        raise ValueError("query_count must select at least one node in the graph")
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data)[-query_count:]
    return embeddings.cpu().numpy()


def _canonical_edges(edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return edge_index
    keys = edge_index[0] * node_count + edge_index[1]
    return edge_index[:, torch.argsort(keys)]
