import numpy as np
import pytest

from src.graph.knn_graph import build_inductive_knn_edge_index
from src.preprocessing.graph_scaling import GraphFeatureScaler


def test_inductive_graph_queries_only_training_nodes():
    train_features = np.array([[0.0], [2.0], [10.0]])
    query_features = np.array([[1.8], [8.0]])

    edge_index = build_inductive_knn_edge_index(
        train_features, query_features, k=1
    )

    np.testing.assert_array_equal(edge_index, np.array([[1, 2], [3, 4]]))


def test_inductive_graph_caps_neighbors_and_validates_query_features():
    train_features = np.array([[0.0], [1.0]])
    query_features = np.array([[0.5]])

    edge_index = build_inductive_knn_edge_index(
        train_features, query_features, k=5
    )

    np.testing.assert_array_equal(edge_index, np.array([[0, 1], [2, 2]]))
    with pytest.raises(ValueError, match="same number"):
        build_inductive_knn_edge_index(train_features, np.ones((1, 2)), k=1)


def test_graph_scaler_fits_only_on_training_nodes():
    train_features = np.array([[0.0, 100.0], [2.0, 100.0]])
    query_features = np.array([[4.0, 100.0]])
    scaler = GraphFeatureScaler()

    train_scaled = scaler.fit_transform(train_features)
    query_scaled = scaler.transform(query_features)

    np.testing.assert_allclose(train_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(query_scaled, [[3.0, 0.0]], atol=1e-7)
    assert scaler.transform(np.empty((0, 2))).shape == (0, 2)
