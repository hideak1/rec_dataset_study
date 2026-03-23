# Getting Started

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager
- Kaggle API credentials (for dataset downloads)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd rec_system_experimental

# Install dependencies with uv
make install

# Download datasets
make download-data

# Start servers
make serve
```

## Kaggle API Setup

1. Go to [Kaggle Account](https://www.kaggle.com/settings) and create an API token
2. Place `kaggle.json` in `~/.kaggle/`
3. Run `chmod 600 ~/.kaggle/kaggle.json`

## GPU Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
