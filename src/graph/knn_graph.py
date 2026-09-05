"""kNN graph construction helpers for customer feature tables."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_edge_index(
    features: np.ndarray,
    k: int = 10,
    use_faiss: bool = False,
    include_self: bool = False,
) -> np.ndarray:
    """Build a sparse directed edge list from tabular customer features.

    Returns a NumPy array of shape (2, num_edges), compatible with PyG edge_index.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    num_nodes = features.shape[0]
    neighbor_count = k if include_self else k + 1

    if use_faiss:
        try:
            import faiss

            index = faiss.IndexFlatL2(features.shape[1])
            index.add(features.astype(np.float32))
            _, indices = index.search(features.astype(np.float32), neighbor_count)
        except ImportError:
            indices = _knn_with_sklearn(features, neighbor_count)
    else:
        indices = _knn_with_sklearn(features, neighbor_count)

    src_nodes = np.repeat(np.arange(num_nodes), neighbor_count)
    dst_nodes = indices.reshape(-1)

    if not include_self:
        mask = src_nodes != dst_nodes
        src_nodes = src_nodes[mask]
        dst_nodes = dst_nodes[mask]

    return np.vstack([src_nodes, dst_nodes]).astype(np.int64)


def _knn_with_sklearn(features: np.ndarray, neighbor_count: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    model.fit(features)
    _, indices = model.kneighbors(features)
    return indices
