# RecSys Experimental Tutorials

Real-world dataset experiments for recommendation systems, covering feature crossing, sequential modeling, and CVR prediction.

## Experiments

| Experiment | Dataset | Focus | Models |
|-----------|---------|-------|--------|
| 1 | Criteo (CTR) | Feature Crossing | DeepFM, DCN, DCN-V2 |
| 2 | Taobao User Behavior | Sequential Modeling | DIN, BST, SASRec |
| 3 | Ali-CCP | CVR Prediction | ESMM, MMoE, PLE |

## Quick Start

```bash
# Install dependencies
make install

# Download datasets
make download-data

# Start tutorial servers
make serve
```

## Environment

- **Package manager**: uv
- **Deep learning**: PyTorch (GPU)
- **Python**: 3.10 - 3.12
