# Recommendation System Experiments

Hands-on tutorial covering three core recommendation system paradigms with real-world datasets and GPU training.

## Experiments

| # | Dataset | Focus | Models | Key Metric |
|---|---------|-------|--------|------------|
| 1 | Criteo (1M subsample) | Feature Crossing | DeepFM, DCN-V1, DCN-V2 | AUC: 0.79 (DCN-V2) |
| 2 | Taobao User Behavior | Sequential Modeling | DIN, DIEN, BST, SASRec | AUC: 0.85 (DIEN) |
| 3 | Tenrec | Multi-Task CVR | ESMM, MMoE, PLE | AUC: 0.69 (ESMM) |

## Setup

```bash
# Create environment with uv
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (with CUDA PyTorch)
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Download datasets
python scripts/download_criteo.py
python scripts/download_taobao.py
# Tenrec: place in data/Tenrec/ manually
```

## Structure

```
notebooks/
  01_criteo_feature_crossing/   # EDA, DeepFM, DCN, Comparison
  02_taobao_sequential/         # EDA, DIN, BST+SASRec, Comparison
  03_tenrec_cvr/                # EDA, ESMM, MMoE+PLE, Comparison
scripts/                        # Standalone training scripts (GPU)
data/                           # Downloaded datasets
```

Each experiment follows a 4-notebook arc: Data Exploration, Model A, Model B, Comparison with statistical tests.

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with 8+ GB VRAM (tested on RTX 3060 12GB)
- ~50GB disk space for datasets
