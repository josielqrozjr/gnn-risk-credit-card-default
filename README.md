# GNN for Credit Card Default Prediction

This repository contains the final empirical implementation for credit-card
default prediction using temporal customer features, similarity graphs,
Graph Neural Networks (GNNs), and a tabular-relational hybrid.

The study uses the public American Express Default Prediction training data:

- 5,531,451 monthly records;
- 458,913 customers;
- 190 columns, including `customer_ID`, `S_2`, and model attributes;
- partitioned customer labels in `data/raw/parquet/train_labels/`.

## Final status

The LightGBM baseline was executed on all 458,913 customers. The GNN
experiments were executed with the customer-disjoint inductive protocol at
25% and 50% of the customer population. The evaluated temporal views are
3M, 6M, and 13M, and the evaluated architectures are GCN, GraphSAGE, and
GAT.

GraphSAGE was the strongest GNN, but LightGBM remained superior:

| Sample | Model | AMEX | ROC-AUC |
|---:|---|---:|---:|
| 25% | LightGBM | **0.7798** | **0.9573** |
| 25% | Best GNN (GraphSAGE, 6M) | 0.7058 | 0.9343 |
| 50% | LightGBM | **0.7819** | **0.9591** |
| 50% | Best GNN (GraphSAGE, 13M) | 0.7084 | 0.9399 |

The conclusion supported by the completed experiments is that relational
message passing did not add predictive value over the tabular representation
under this protocol. The 50% Hybrid OOF run was anomalous under GPU memory
pressure and is excluded from the conclusion.

The complete results and limitations are documented in
[RELATORIO_RESULTADOS.md](RELATORIO_RESULTADOS.md).

## Pipeline

```text
AMEX Parquet data
  -> schema and integrity validation
  -> sort by customer_ID and S_2
  -> temporal features: 3M, 6M, 13M
  -> customer-level stratified split: 64% train / 16% validation / 20% test
  -> LightGBM tabular baseline
  -> training-only kNN graph with k=10
  -> GCN / GraphSAGE / GAT
  -> frozen inductive embeddings for validation and test customers
  -> metrics and sanitized artifacts
```

The graph topology is target-independent. Scaling is fitted on training
customers, and held-out customers are queried against the training graph.
The evaluation reports AMEX Score, ROC-AUC, PR-AUC, Brier Score, and ECE.

## Hybrid OOF

The implemented hybrid uses three out-of-fold stages:

```text
GNN training folds
  -> out-of-fold relational embeddings
  -> concatenate tabular features and embeddings
  -> second-stage LightGBM classifier
```

The 25% hybrid result was plausible but below the baseline. The 50% result
was anomalous (AMEX -0.2297 and ROC-AUC 0.2439) after CUDA allocations of
approximately 10–11 GB on a 6 GB GPU. It must not be used as a scientific
result until the memory behavior is corrected and the run is repeated.

## Reproduction

### Tabular environment

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.baseline.run --config configs/baseline.json
```

### Complete experiment runner

```bash
python3 -m scripts.run_full_experiments --dry-run
python3 -m scripts.run_full_experiments
```

The full runner preserves the configured feature width and requires adequate
RAM for full-customer graph phases. Its logs are written to `logs/`, and its
outputs are written to `artifacts/baseline/full/` and
`artifacts/experiments/full/`.

### Windows and CUDA

```powershell
py -3.13 -m venv C:\gnn-venv
C:\gnn-venv\Scripts\python.exe -m pip install -r requirements-gnn.txt
C:\gnn-venv\Scripts\python.exe -m scripts.run_full_experiments --device cuda --sample-customers 229457
```

Use `--device cpu` to force CPU execution or `--device auto` to select CUDA
when available. FAISS is used for memory-bounded CPU kNN construction. The
LightGBM baseline and the hybrid's second-stage classifier run on the CPU.

## Source modules and artifacts

The executable implementation is organized as follows:

```text
configs/baseline.json
scripts/run_full_experiments.py
scripts/run_real_experiments.py
scripts/run_hybrid_oof.py
src/
  baseline/
  data/
  evaluation/
  experiments/
  graph/
  models/
  preprocessing/
  reporting/
  temporal_features/
tests/
artifacts/
  baseline/full/
  experiments/25pct/
  experiments/50pct/
overleaf/
  project_planning.tex
  lang/pt-br_project_planning.tex
  assets/diagram-complete.svg
```

The repository also retains `src/mvp/` and `artifacts/mvp/` for compatibility
with historical smoke and bounded-result commands. They are not the current
project scope.

Tracked artifacts are sanitized. Raw records, customer identifiers, row-level
predictions, and large local datasets are not committed.

## Explicit final limitations

- The GNN protocol was stable and reported up to 50% of the customer
  population; no 100%-customer full-batch GNN result is claimed.
- The anomalous 50% Hybrid OOF run is excluded from the conclusion.
- An operational 0–1000 calibrated Risk Score is not implemented.
- Individual SHAP, GNNExplainer, and influential-neighbor explanations are
  not implemented. LightGBM feature importance is available.
- Formal multi-seed, variable-k, similarity-function, and attention-fusion
  ablation batteries are not reported as completed experiments.

These are final scope limitations, not a backlog. The project is
closed at the empirical scope described above.
