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

    requested_neighbors = min(k, num_nodes) if include_self else min(k + 1, num_nodes)
    neighbor_count = min(requested_neighbors, num_nodes)
    target_neighbors = min(k, num_nodes) if include_self else min(k, max(num_nodes - 1, 0))
    indices = _search_neighbors(features, neighbor_count, use_faiss)
    fallback_indices = None

    src_nodes_list: list[int] = []
    dst_nodes_list: list[int] = []
    for src, row in enumerate(indices):
        seen: set[int] = set()
        candidate_rows = [row]
        if not include_self and len(row) < num_nodes:
            if fallback_indices is None:
                fallback_indices = _search_neighbors(features, num_nodes, use_faiss)
            candidate_rows.append(fallback_indices[src])

        for candidates in candidate_rows:
            for dst in candidates:
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
            if len(seen) >= target_neighbors:
                break

    return np.vstack([src_nodes_list, dst_nodes_list]).astype(np.int64)


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
