# GNN for Credit Default Prediction: Relational and Temporal Hypothesis

Starter repository for machine learning research on credit default prediction using temporal feature engineering, relational graph construction, and graph neural network representations.

## Methodology

1. **Temporal Feature Engineering**
   - Aggregate monthly customer time-series into static features (mean, std, last, and trend/diff).
2. **kNN Graph Construction**
   - Build a customer graph from tabular features using scalable nearest-neighbor search.
3. **GAT-based Representation Learning**
   - Train graph attention networks to learn node embeddings for default-risk prediction.
4. **Hybrid LightGBM Model**
   - Concatenate original tabular features with GNN embeddings and train a boosted-tree classifier.

## Project Structure

`data/`, `experiments/`, and `results/` are intentionally gitignored and should be created/used locally for datasets and outputs.

```text
.
├── data/
│   ├── raw/
│   ├── parquet/
│   ├── temporal/
│   ├── customer_features/
│   └── graphs/
│       ├── train/
│       └── test/
│
├── experiments/
├── results/
│
├── src/
│   ├── preprocessing/
│   ├── temporal_features/
│   │   └── aggregation.py
│   ├── graph/
│   │   ├── similarity.py
│   │   ├── knn_graph.py
│   │   └── graph_validation.py
│   ├── models/
│   │   ├── gcn.py
│   │   ├── graphsage.py
│   │   ├── gat.py
│   │   └── hybrid.py
│   ├── risk/
│   │   ├── calibration.py
│   │   └── scoring.py
│   ├── explainability/
│   │   ├── gnn_explainer.py
│   │   └── feature_analysis.py
│   └── evaluation/
│       ├── metrics.py
│       ├── ablation.py
│       └── robustness.py
│
├── requirements.txt
└── README.md
```
