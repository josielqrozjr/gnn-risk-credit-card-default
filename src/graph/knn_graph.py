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
    if not np.isfinite(features).all():
        raise ValueError("features must only contain finite values")

    if k < 1:
        raise ValueError("k must be >= 1")

    num_nodes = features.shape[0]
    if num_nodes == 0:
        return np.empty((2, 0), dtype=np.int64)

    requested_neighbors = min(k, num_nodes) if include_self else min(k + 1, num_nodes)
    neighbor_count = min(requested_neighbors, num_nodes)
    target_neighbors = min(k, num_nodes) if include_self else min(k, max(num_nodes - 1, 0))
    indices = _search_neighbors(features, neighbor_count, use_faiss)

    src_nodes_list: list[int] = []
    dst_nodes_list: list[int] = []
    for src, row in enumerate(indices):
        seen: set[int] = set()
        for dst in row:
            dst_int = int(dst)
            if not include_self and dst_int == src:
                continue
            if dst_int in seen:
                continue
            seen.add(dst_int)
            src_nodes_list.append(src)
            dst_nodes_list.append(dst_int)
            if len(seen) >= target_neighbors:
                break

    if not src_nodes_list:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([src_nodes_list, dst_nodes_list]).astype(np.int64)


def build_inductive_knn_edge_index(
    train_features: np.ndarray,
    query_features: np.ndarray,
    k: int = 10,
    use_faiss: bool = False,
) -> np.ndarray:
    """Connect held-out query nodes to a training-only neighbor index.

    The returned PyG edge index contains edges from training nodes to query nodes.
    Query node indices are offset by the number of training nodes so the result
    can be appended to the training edge index.
    """
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("train_features and query_features must be 2D arrays")
    if train_features.shape[1] == 0:
        raise ValueError("feature matrices must contain at least one feature")
    if train_features.shape[1] != query_features.shape[1]:
        raise ValueError("train and query features must have the same number of columns")
    if not np.isfinite(train_features).all() or not np.isfinite(query_features).all():
        raise ValueError("features must only contain finite values")
    if k < 1:
        raise ValueError("k must be >= 1")
    if query_features.shape[0] == 0:
        return np.empty((2, 0), dtype=np.int64)
    if train_features.shape[0] == 0:
        raise ValueError("at least one training node is required for query nodes")

    neighbor_count = min(k, train_features.shape[0])
    indices = _search_query_neighbors(
        train_features, query_features, neighbor_count, use_faiss
    )
    train_nodes = indices.reshape(-1)
    query_nodes = np.repeat(
        np.arange(query_features.shape[0], dtype=np.int64) + train_features.shape[0],
        neighbor_count,
    )
    return np.vstack([train_nodes, query_nodes])


def _search_neighbors(features: np.ndarray, neighbor_count: int, use_faiss: bool) -> np.ndarray:
    if use_faiss:
        try:
            import faiss

            features_f32 = np.ascontiguousarray(features, dtype=np.float32)
            index = faiss.IndexFlatL2(features_f32.shape[1])
            index.add(features_f32)
            _, indices = index.search(features_f32, neighbor_count)
            return indices
        except ImportError:
            return _knn_with_sklearn(features, neighbor_count)

    return _knn_with_sklearn(features, neighbor_count)


def _knn_with_sklearn(features: np.ndarray, neighbor_count: int) -> np.ndarray:
    if neighbor_count == 0:
        return np.empty((features.shape[0], 0), dtype=np.int64)
    model = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    model.fit(features)
    _, indices = model.kneighbors(features)
    return indices


def _search_query_neighbors(
    train_features: np.ndarray,
    query_features: np.ndarray,
    neighbor_count: int,
    use_faiss: bool,
) -> np.ndarray:
    if use_faiss:
        try:
            import faiss

            index_features = np.ascontiguousarray(train_features, dtype=np.float32)
            query_features_f32 = np.ascontiguousarray(query_features, dtype=np.float32)
            index = faiss.IndexFlatL2(index_features.shape[1])
            index.add(index_features)
            _, indices = index.search(query_features_f32, neighbor_count)
            return indices
        except ImportError:
            pass

    model = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    model.fit(train_features)
    _, indices = model.kneighbors(query_features)
    return indices
