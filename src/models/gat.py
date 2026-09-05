"""Starter Graph Attention Network (GAT) model for node classification."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    """Two-layer GAT for binary/multi-class node-level prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run a forward pass given a PyG data object with x and edge_index."""
        x, edge_index = data.x, data.edge_index
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = nn.functional.elu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        logits = self.gat2(x, edge_index)
        return logits
