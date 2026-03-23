# Experiment 2: Taobao — Sequential Modeling

The [Taobao User Behavior](https://www.kaggle.com/datasets/marwa80/userbehavior) dataset contains ~100 million user behavior records from Taobao, including click, purchase, add-to-cart, and favorite actions.

## Focus: Sequential & Attention-based Models

| Notebook | Model | Key Idea |
|----------|-------|----------|
| 2.1 | — | Dataset exploration, behavior sequence analysis, temporal patterns |
| 2.2 | DIN | Target-aware attention over user behavior sequences |
| 2.3 | BST / SASRec | Transformer self-attention for sequential recommendation |
| 2.4 | — | Model comparison, attention visualization, ablation studies |

## Target Performance

| Model | AUC |
|-------|-----|
| DIN | ~0.63+ |
| BST | ~0.64+ |
| SASRec (next-item) | HR@10 ~0.50+ |
