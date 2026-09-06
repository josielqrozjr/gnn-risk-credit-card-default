# GNN for Credit Card Default Prediction Using Relational and Temporal Features

This repository supports a machine learning research project on credit card default prediction using temporal feature engineering, customer-similarity graphs, Graph Neural Networks (GNNs), and hybrid tabular-relational models. The study uses the public **American Express Default Prediction (AMEX)** dataset.

The project tests whether a customer's financial behavior contains additional relational information when compared with the behavior of similar customers. Instead of representing each customer exclusively as an independent feature vector, the proposed method builds a graph in which customers are nodes and behavioral similarity defines the edges.

The second dimension of the study is temporal. Each customer's available financial history is represented over different observation horizons to determine whether relationships derived from recent, intermediate, and long-term behavior have different risk-discrimination capabilities.

The methodology combines three main components:

1. **Behavioral representation:** transformation of each customer's monthly history into statistical and temporal features.
2. **Relational representation:** construction of customer-similarity graphs.
3. **Graph learning:** use of GCN, GraphSAGE, and GAT to learn relational representations and classify customers according to their probability of default.

As analytical extensions, the study investigates a **calibrated Risk Score** and the **explainability of GNN decisions**. The scope focuses exclusively on the supervised relational-temporal core and excludes nonessential complexity such as reinforcement learning.

The central hypothesis is that **customer relational structure, combined with temporal information, provides additional predictive information that independent tabular models do not capture**.

## 1. Research Questions

### Main Research Question

> **Does the relational and temporal representation of customers through Graph Neural Networks improve credit card default detection compared with traditional tabular models?**

### Secondary Research Questions

1. Does constructing relationships among customers based on behavioral similarity add predictive information beyond individual attributes?
2. Do different temporal horizons produce relational structures with different default-detection capabilities?
3. Do GAT, GraphSAGE, and GCN behave differently when representing risk relationships among customers?
4. Does combining relational embeddings with tabular features improve performance compared with using the GNN alone?
5. Can the probabilities produced by the GNN be transformed into a calibrated and interpretable Risk Score?
6. Can model explanations identify which temporal features and customer relationships contributed to each decision?

## 2. Hypotheses

### H1: Relational Hypothesis

> Customers with similar financial and behavioral characteristics may exhibit related risk patterns, and using these relationships through a GNN improves default detection compared with models that treat each customer as an independent observation.

### H2: Temporal Hypothesis

> Representing customers over different temporal horizons captures different manifestations of financial behavior and produces complementary information for default prediction.

### H3: Hybrid Hypothesis

> Embeddings produced by the GNN may contain relational information complementary to tabular features, so combining both representations may outperform either representation used alone.

### H4: Explainability Hypothesis

> Joint analysis of feature contributions and customer relationships may provide more informative explanations of individual risk than analysis based exclusively on tabular feature importance.

## 3. Dataset

The study will use only the training portion of the **American Express Default Prediction** competition, comprising `train_data.csv` and `train_labels.csv`. The competition's public test set will not be used to evaluate this study.

The experimental dataset contains approximately:

- **5,531,451 temporal records**;
- **458,913 unique customers**;
- approximately **190 original attributes**;
- up to 13 monthly observations per customer;
- a binary `target` associated with each `customer_ID`.

Variables are grouped into the following families:

- `D_*`: *Delinquency*;
- `S_*`: *Spend*;
- `P_*`: *Payment*;
- `B_*`: *Balance*;
- `R_*`: *Risk*.

Predefined categorical variables:

```text
B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_66, D_68
```

The `customer_ID` will be used exclusively as an identifier and relational key.

## 4. Experimental Integrity Principles

1. **Split unit:** Customers will first be divided into an 80% development portion and a 20% held-out test portion. The validation fold will be drawn only from the development portion. All train, validation, and test partitions will be strictly customer-disjoint by `customer_ID`, preventing temporal or customer-identity leakage across folds.
2. **Use of the target:** The `target` will be used exclusively in the supervised loss function. It will never be used to calculate node similarity, create edges, or induce graph topology.
3. **Feature selection and normalization:** Any fitted statistical transformation will be learned strictly from the training fold and projected onto validation and test data.
4. **Inductive protocol:** The pipeline will prioritize inductive inference to simulate the arrival of new customers without rebuilding the complete existing topology.

Because the study evaluates generalization to unseen customers, its primary protocol follows this inductive sequence:

```text
Training data
  ↓
Training graph
  ↓
GNN training
  ↓
Frozen encoder
  ↓
New validation/test customers
  ↓
Construction of relationships allowed by the protocol
  ↓
Inference
```

Test labels will not be used to construct representations. A transductive evaluation may be conducted only as an additional experiment that is clearly identified and separated from the primary result.

## 5. Methodology Architecture

```mermaid
flowchart TD
    A[AMEX training data<br/>5,531,451 records<br/>458,913 customers] --> B[Schema and integrity validation]
    B --> C[Sort by customer_ID and S_2]
    C --> D[Multiscale temporal feature engineering]
    D --> D1[Differences and deltas]
    D --> D2[Temporal statistics]
    D --> D3[3M / 6M / 13M windows]
    D1 --> E[Customer-level behavioral representation]
    D2 --> E
    D3 --> E
    E --> F[Customer-level 80/20 partitioning]
    F --> G[Development / training set]
    F --> V[Validation set]
    F --> H[Held-out test set]
    G --> I[Training kNN graph construction]
    I --> J[GCN / GraphSAGE / GAT]
    J --> K[Relational embedding]
    G --> L[LightGBM tabular baseline]
    K --> M[Trained classifier / hybrid fusion]
    L --> M
    V --> S[Model selection and calibration]
    M --> S
    H --> R[Test inference]
    S --> R
    R --> N[Risk Score / default probability]
    N --> O[Metrics and calibration analysis]
    O --> P[Ablation and robustness]
    P --> Q[Explainability]
```

## 6. Multiscale Temporal Feature Engineering

Each customer's monthly temporal dimension will be condensed into static and dynamic features over the following windows:

- **Short term (3 months):** captures the most recent financial stress.
- **Medium term (6 months):** captures transitions in credit behavior.
- **Long term (13 months):** captures complete history and customer stability.

### Temporal Operations

- Successive differences: $\Delta X_t = X_t - X_{t-1}$.
- Descriptive statistics: mean, standard deviation, minimum, maximum, first record, and last record.
- Trends and slopes: $\mathrm{total\_delta} = X_{last} - X_{first}$ and $\mathrm{avg\_monthly\_slope} = \frac{X_{last} - X_{first}}{n - 1}$.
- Volatility and categorical-state dynamics, including the number of transitions and the last state.

The existing `aggregate_temporal_features` utility currently computes the mean, standard deviation, last value, and first-to-last difference for each selected numerical attribute. The remaining planned statistics, categorical transitions, and explicit 3M/6M/13M orchestration will extend this module.

## 7. Behavioral Graph Construction and Scalability

- **Nodes ($V$):** customers ($N \approx 458{,}913$).
- **Edges ($E$):** behavioral-similarity relationships calculated with $k$-Nearest Neighbors ($k$-NN) over normalized features.
- **Multiview topology:** three independent sparse graphs, $G_{3M}$, $G_{6M}$, and $G_{13M}$.
- **M1 scalability and out-of-memory prevention:**
  - Approximate Nearest Neighbors, such as Annoy, HNSW, or FAISS, to avoid $O(N^2)$ pairwise construction.
  - Sparse edge-list storage through `torch_geometric.data.Data(edge_index=...)`, without materializing dense adjacency matrices.
  - Mini-batching through neighbor sampling with `NeighborLoader`.

The current `build_knn_edge_index` implementation constructs a sparse directed edge list using scikit-learn or FAISS, validates finite two-dimensional inputs, and can include or exclude self-loops. Approximate indexing, cosine distance, explicit train-to-held-out queries, and graph-validation utilities remain planned extensions.

## 8. Models, Multiview Fusion, and Hybrid Approach

### 8.1 Compared GNNs

- **GCN:** baseline with isotropic convolutional aggregation.
- **GraphSAGE:** efficient inductive neighbor sampling and aggregation.
- **GAT:** adaptive attention coefficients assigned to behavioral edges.

At least these three GNN families will be evaluated.

#### GCN

The Graph Convolutional Network will be used as the graph-convolution baseline. Its role is to determine whether simple neighborhood information aggregation already provides a gain over the tabular model.

#### GraphSAGE

GraphSAGE will be used as an alternative based on neighbor aggregation and node-embedding generation, enabling a configuration more compatible with inference for new customers.

#### GAT

The Graph Attention Network is the primary candidate architecture. GAT assigns different weights to relationships between a node and its neighbors, allowing the study to investigate whether some behavioral relationships are more relevant than others.

The comparison will assess which mechanism best represents customer risk relationships:

```text
GCN
 ↓
GraphSAGE
 ↓
GAT
```

The repository currently includes a two-layer `GATClassifier` for node-level prediction. GCN and GraphSAGE implementations remain planned.

### 8.2 Multiview Fusion

Concatenation and an attention mechanism will combine the three temporal-view representations, $H_{3M}$, $H_{6M}$, and $H_{13M}$.

### 8.3 Hybrid Model

Relational embeddings learned by the GNN will be joined with the original tabular vectors and provided to a Gradient Boosting classifier such as LightGBM. The current `HybridLightGBMClassifier` concatenates tabular features and GNN embeddings for fitting and prediction. The final experimental pipeline will use out-of-fold GNN representations when training the second-stage classifier.

## 9. Ablation Studies

The ablation study is a central component of the scientific contribution because it separates the effect of each part of the methodology. Every comparison will preserve the same customer split, preprocessing protocol, and, where applicable, training budget. Consequently, performance differences can be attributed primarily to the component under evaluation.

Configurations will be selected using the training and validation sets. The held-out test set will be used only for the final evaluation of the selected configurations, preventing indirect adaptation to test results.

The main metrics will be AMEX Score, ROC-AUC, PR-AUC, F1 Score, Brier Score, and Expected Calibration Error (ECE). Because default prediction is imbalanced, PR-AUC will receive priority in complementary discrimination analysis, while Brier Score and ECE will evaluate probability quality. The official AMEX Score remains the primary competition-aligned metric.

### 9.1 Ablation 1: Relationships

```text
Complete model
  ↓
Remove edges
  ↓
Attribute-only model
```

In this comparison, the complete model uses customer attributes and the k-NN neighborhood structure. In the no-edge condition, the same individual representation is preserved, but message passing or aggregation among customers is removed.

**Objective:** verify the specific gain from relational structure and test whether customer relationships provide predictive information beyond individual attributes.

An improvement of the complete model over the no-edge model will support the relational hypothesis. Representation dimensionality, classifier, and other hyperparameters should remain as equivalent as possible.

### 9.2 Ablation 2: Temporality

```text
13M
 ↓
6M
 ↓
3M
 ↓
3M + 6M + 13M
```

Representations built from the complete 13-month history, the intermediate 6-month window, the recent 3-month window, and the combination of all three windows will be compared. Each configuration must generate its own temporal features and, where applicable, its own similarity graph.

**Objective:** identify the contribution of each temporal horizon and determine whether combining short-, medium-, and long-term information provides complementary information.

In addition to mean performance, calibration and stability differences will be analyzed. For example, the 3-month window may capture recent financial stress, whereas the 13-month window may better represent a customer's historical stability.

### 9.3 Ablation 3: GNN Architecture

```text
GCN
 ↓
GraphSAGE
 ↓
GAT
```

The architectures will be trained on the same node representation and topology whenever the architecture permits. The number of layers, embedding dimension, optimizer, and stopping criterion will be controlled to make the comparison fairer.

**Objective:** evaluate the effect of the aggregation mechanism on risk classification.

GCN represents more uniform convolutional aggregation, GraphSAGE emphasizes inductive aggregation, and GAT assigns different attention weights to relationships. Training time, memory consumption, and parameter count will also be recorded because the best model will not be determined solely by a predictive metric.

### 9.4 Ablation 4: Fusion

```text
Concatenation vs. attention fusion
```

With concatenation, embeddings produced by the temporal views are joined directly before classification. With attention fusion, the model learns weights that combine the 3M, 6M, and 13M representations according to their relevance to each customer or task.

**Objective:** determine whether attention across temporal horizons provides a real gain over simple fusion.

In addition to performance, attention weights may be analyzed to determine whether the model consistently prioritizes one temporal window or uses different horizons for different customer profiles. This interpretation will be treated as descriptive evidence, not causality.

### 9.5 Ablation 5: Hybrid Representation

```text
Tabular
   vs.
GNN
   vs.
Tabular + GNN
```

Three representations will be compared for the final classifier:

1. **Tabular:** only the customer's behavioral and temporal features.
2. **GNN:** only the relational embedding learned by the GNN.
3. **Tabular + GNN:** a combination of tabular features and the relational embedding.

**Objective:** investigate whether the relational representation contains complementary information and whether hybrid fusion outperforms either source used alone.

To prevent vector size alone from influencing the comparison, representation dimensions will be documented and classifiers of comparable complexity will be used when necessary. The most relevant result will be the difference between the hybrid model and the two standalone models.

### 9.6 Ablation 6: Graph Structure

Different values of `k`, including `k = 5`, `k = 10`, and `k = 20`, will be evaluated to observe the effect of neighborhood density on performance, memory, and processing time. If computationally feasible, different similarity functions, such as Euclidean distance and cosine similarity, will also be compared.

**Objective:** determine whether the results depend on a specific topology choice and identify a stable range of configurations.

The value of `k`, the similarity function, and any associated normalization will be defined without using `target` to create edges. The comparisons will also determine whether denser graphs increase cost without producing consistent predictive gains.

### 9.7 Comparison Protocol

For each ablation, the following will be recorded:

- complete configuration and the component removed or changed;
- data split and random seed;
- number of parameters, training time, and memory usage when measurable;
- AMEX Score, ROC-AUC, PR-AUC, F1 Score, Brier Score, and ECE;
- absolute difference from the complete model;
- variation across repetitions when multiple runs are available.

Conclusions will be based on consistent differences among configurations rather than on a single metric or execution. When a difference is small, the component's contribution will be reported as limited or inconclusive.

## 10. Robustness Study

The robustness study will determine whether the conclusions remain similar when reasonable experimental conditions change. Its purpose is to distinguish a structural methodological gain from a result dependent on a random seed, a value of `k`, or a particular architecture choice.

Robustness will be evaluated across:

- different random seeds;
- different values of `k`;
- different GNN architectures;
- different similarity functions;
- different temporal horizons;
- different fusion strategies.

Whenever possible, multiple independent repetitions will be performed while preserving the customer split. For each configuration, the mean and standard deviation of the metrics will be calculated:

$$
\mathrm{result} = \mathrm{mean} \pm \mathrm{std}
$$

The worst and best results across repetitions will also be observed, especially for PR-AUC, Brier Score, and ECE. A configuration will be considered more robust when it maintains competitive performance with low variability and without excessive degradation under alternative conditions.

Robustness analyses will not replace the final evaluation on the held-out test set. They will be conducted using training and validation data; after the final configuration is selected, it will be evaluated once on the reserved test set. This keeps model selection separate from the final generalization estimate.

## 11. Calibration, Risk Score, and Explainability

- **Risk Score:** monotonic mapping of estimated default probabilities ($PD \in [0,1]$) to an interpretable credit scale from 0 to 1000.
- **Calibration:** evaluation with Brier Score and Expected Calibration Error (ECE), using Platt Scaling or Isotonic Regression fitted on the validation fold.
- **Explainability:**
  - **Global:** importance of tabular features through SHAP.
  - **Relational and local:** identification of influential neighbors and critical subgraphs through GAT attention weights and GNNExplainer.

## 12. Research Schedule

The project schedule covers ten weeks, beginning with the submission of the proposal and ending on October 31.

| Activity | Weeks 1--2 | Weeks 3--4 | Weeks 5--6 | Weeks 7--8 | Weeks 9--10 (until 10/31) |
|---|:---:|:---:|:---:|:---:|:---:|
| State-of-the-art survey | X |  |  |  |  |
| Dataset processing | X | X |  |  |  |
| Experiment 1: tabular baselines and pilot graphs |  | X | X |  |  |
| Experiment 2: GNN architectures: GCN, GraphSAGE, GAT |  |  | X |  |  |
| Experiment 3: multiview, hybrid model, and calibration |  |  | X | X |  |
| Results analysis and explainability |  |  | X | X |  |
| Manuscript writing |  |  | X | X | X |

## 13. Repository and Source Code

The repository separates reusable source modules from local datasets, experimental scripts, generated artifacts, and research documents. The codebase is currently a starter implementation: four source modules exist, while the remaining modules shown below define the intended final architecture.

### 13.1 Current Source Modules

- `src/temporal_features/aggregation.py`: aggregates customer time series into customer-level mean, standard deviation, last-value, and difference features.
- `src/graph/knn_graph.py`: validates feature matrices and builds sparse directed k-NN edge indices with scikit-learn or FAISS.
- `src/models/gat.py`: provides a two-layer PyTorch Geometric GAT classifier for node-level prediction.
- `src/models/hybrid.py`: concatenates tabular features with GNN embeddings and trains a LightGBM classifier.
- `requirements.txt`: declares the Python libraries used by the planned pipeline.
- `overleaf/`: existing local directory reserved for the research article and related publication assets.

### 13.2 Current and Planned Directory Structure

The markers below distinguish the current filesystem from the intended architecture:

- `[existing]`: currently present in the repository workspace.
- `[planned]`: to be created as implementation progresses.
- `[local]`: generated or dataset content intended to remain outside version control.

```text
.
├── .gitignore                                      [existing]
├── LICENSE                                         [existing]
├── README.md                                       [existing]
├── requirements.txt                                [existing]
├── overleaf/                                       [existing]
│   └── publication sources and assets              [planned]
├── data/                                           [planned, local]
│   ├── raw/                                        [planned, local]
│   │   ├── train_data.csv                          [planned, local]
│   │   └── train_labels.csv                        [planned, local]
│   ├── parquet/                                    [planned, local]
│   ├── temporal/                                   [planned, local]
│   ├── customer_features/                          [planned, local]
│   └── graphs/                                     [planned, local]
│       ├── train/                                  [planned, local]
│       ├── validation/                             [planned, local]
│       └── test/                                   [planned, local]
├── experiments/                                    [planned, local]
│   ├── baselines/                                  [planned, local]
│   ├── gnn/                                        [planned, local]
│   ├── fusion/                                     [planned, local]
│   ├── ablation/                                   [planned, local]
│   └── robustness/                                 [planned, local]
├── results/                                        [planned, local]
│   ├── metrics/                                    [planned, local]
│   ├── figures/                                    [planned, local]
│   ├── explanations/                               [planned, local]
│   └── models/                                     [planned, local]
└── src/                                            [existing]
    ├── preprocessing/                              [planned]
    │   ├── data_validation.py                      [planned]
    │   ├── splitting.py                            [planned]
    │   └── transformations.py                      [planned]
    ├── temporal_features/                          [existing]
    │   └── aggregation.py                          [existing]
    ├── graph/                                      [existing]
    │   ├── similarity.py                           [planned]
    │   ├── knn_graph.py                            [existing]
    │   └── graph_validation.py                     [planned]
    ├── models/                                     [existing]
    │   ├── gcn.py                                  [planned]
    │   ├── graphsage.py                            [planned]
    │   ├── gat.py                                  [existing]
    │   ├── multiview.py                            [planned]
    │   └── hybrid.py                               [existing]
    ├── risk/                                       [planned]
    │   ├── calibration.py                          [planned]
    │   └── scoring.py                              [planned]
    ├── explainability/                             [planned]
    │   ├── gnn_explainer.py                        [planned]
    │   └── feature_analysis.py                     [planned]
    └── evaluation/                                 [planned]
        ├── metrics.py                              [planned]
        ├── ablation.py                             [planned]
        └── robustness.py                           [planned]
```

Local dataset and output directories should not be committed. They must contain only data obtained under the AMEX competition terms, intermediate artifacts, model checkpoints, metrics, figures, and explanations produced by local experiments.

### 13.3 Dependencies

The declared Python dependencies are:

- PyTorch and PyTorch Geometric for GNN training;
- Pandas and Polars for tabular and temporal processing;
- scikit-learn for preprocessing, nearest-neighbor search, calibration, and evaluation;
- LightGBM for tabular and hybrid models;
- SHAP for feature-level explanations;
- FAISS CPU for scalable nearest-neighbor search.

Install them with:

```bash
python -m pip install -r requirements.txt
```

## 14. Final Considerations

This research establishes a rigorous investigation into the information gain produced by relational structures combined with different temporal horizons in the financial domain. By focusing on the supervised core and excluding unnecessarily costly approaches such as reinforcement-learning formulations, the project preserves experimental rigor and remains technically and computationally feasible for the team.

The repository will evolve incrementally from its current starter modules toward the complete pipeline documented above. Every implementation stage should preserve customer-disjoint evaluation, target-independent topology construction, inductive inference, sparse graph processing, reproducible ablations, and a final one-time evaluation on the held-out test set.