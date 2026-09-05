# GNN for Credit Default Prediction: Relational and Temporal Hypothesis

Starter repository for machine learning research on credit default prediction using temporal feature engineering, relational graph construction, and graph neural network representations.

## Methodology

1. **Temporal Feature Engineering**
   - Aggregate monthly customer time-series into static features (mean, std, last, and trend/diff).
2. **kNN Graph Construction**
   - Build a customer graph from tabular features using scalable nearest-neighbor search.
3. **GAT/GraphSAGE-style Representation Learning**
   - Train graph neural networks to learn node embeddings for default-risk prediction.
4. **Hybrid LightGBM Model**
   - Concatenate original tabular features with GNN embeddings and train a boosted-tree classifier.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── parquet/
│   ├── customer_features/
│   └── graphs/
├── experiments/
├── results/
├── src/
│   ├── temporal_features/
│   │   └── aggregation.py
│   ├── graph/
│   │   └── knn_graph.py
│   └── models/
│       ├── gat.py
│       └── hybrid.py
├── requirements.txt
└── README.md
```
