# Experiment 1: Criteo — Feature Crossing

The [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-advertising-challenge) dataset is the standard benchmark for CTR prediction. It contains ~45 million samples with 13 numerical features and 26 categorical features.

## Focus: Feature Interaction Models

| Notebook | Model | Key Idea |
|----------|-------|----------|
| 1.1 | — | Dataset exploration, feature distributions, preprocessing |
| 1.2 | DeepFM | FM for 2nd-order + DNN for high-order, shared embeddings |
| 1.3 | DCN / DCN-V2 | Cross Network for explicit polynomial feature interactions |
| 1.4 | — | Head-to-head comparison, ablation studies, analysis |

## Target Performance

| Model | AUC | LogLoss |
|-------|-----|---------|
| DeepFM | ~0.8007 | ~0.4440 |
| DCN | ~0.8012 | ~0.4435 |
| DCN-V2 | ~0.8026 | ~0.4420 |
