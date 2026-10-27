import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from src.models.gnn import (
    GraphClassifier,
    fit_gnn_classifier,
    predict_gnn_probabilities,
)


@pytest.mark.parametrize("architecture", ["gcn", "graphsage", "gat"])
def test_gnn_classifier_trains_and_predicts_inductive_queries(architecture):
    features = torch.tensor(
        [[0.0, 0.1], [1.0, 0.9], [0.2, 0.3], [0.8, 0.7]],
        dtype=torch.float32,
    )
    train_data = Data(
        x=features[:3],
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    validation_data = Data(
        x=features,
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 0], [1, 0, 2, 1, 3]], dtype=torch.long
        ),
    )
    model = GraphClassifier(
        architecture, in_channels=2, hidden_channels=4, dropout=0.0
    )

    summary = fit_gnn_classifier(
        model,
        train_data,
        torch.tensor([0, 1, 0]),
        validation_data,
        torch.tensor([1]),
        max_epochs=3,
        patience=2,
    )
    probabilities = predict_gnn_probabilities(model, validation_data, query_count=1)

    assert 1 <= summary.best_epoch <= summary.epochs_completed <= 3
    assert 0.0 <= probabilities[0] <= 1.0


def test_gnn_training_rejects_query_to_training_edges():
    features = torch.ones((3, 2), dtype=torch.float32)
    train_data = Data(x=features[:2], edge_index=torch.tensor([[0], [1]]))
    validation_data = Data(
        x=features,
        edge_index=torch.tensor([[0, 2], [1, 0]]),
    )
    model = GraphClassifier("gcn", in_channels=2, hidden_channels=2)

    with pytest.raises(ValueError, match="must not send messages"):
        fit_gnn_classifier(
            model,
            train_data,
            torch.tensor([0, 1]),
            validation_data,
            torch.tensor([1]),
            max_epochs=1,
        )
