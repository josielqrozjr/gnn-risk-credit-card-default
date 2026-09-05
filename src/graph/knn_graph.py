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

    if k < 1:
        raise ValueError("k must be >= 1")

    num_nodes = features.shape[0]
    if num_nodes == 0:
        return np.empty((2, 0), dtype=np.int64)

    requested_neighbors = k if include_self else k + 1
    neighbor_count = min(requested_neighbors, num_nodes)

    if use_faiss:
        try:
            import faiss

            features_f32 = np.ascontiguousarray(features, dtype=np.float32)
            index = faiss.IndexFlatL2(features_f32.shape[1])
            index.add(features_f32)
            _, indices = index.search(features_f32, neighbor_count)
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
