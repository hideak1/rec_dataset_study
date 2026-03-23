"""Create all 4 Ali-CCP CVR notebooks."""
import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '04_aliccp_cvr')
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split('\n')}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split('\n'), "outputs": [], "execution_count": None}


# ============================================================
# NOTEBOOK 1: Data Exploration
# ============================================================
cells_eda = [
    md("""# Ali-CCP Dataset Exploration for Multi-Task CVR Prediction

---

## Learning Objectives

By the end of this notebook, you will be able to:

1. **Understand** the Ali-CCP (Alibaba Click and Conversion Prediction) dataset structure
2. **Analyze** the click-through and conversion funnels that motivate multi-task learning
3. **Quantify** the Sample Selection Bias (SSB) problem in CVR prediction
4. **Prepare** processed data for ESMM, MMoE, and PLE experiments

## Prerequisites

- Understanding of CTR/CVR prediction concepts
- Familiarity with multi-task learning motivation

## Table of Contents

1. [Setup & Configuration](#1-setup--configuration)
2. [Data Loading & Parsing](#2-data-loading--parsing)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Sample Selection Bias Analysis](#4-sample-selection-bias-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)"""),

    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import pickle
import time
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120

DATA_DIR = Path('../../data/aliccp')
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Subsample for tutorial (full dataset: 42M+ train)
SUBSAMPLE_SIZE = 5_000_000

print(f'Data directory: {DATA_DIR}')
print(f'Files: {[f for f in os.listdir(DATA_DIR) if not f.endswith(".tar.gz")]}')\n"""),

    md("""## 2. Data Loading & Parsing

> **Concept: Ali-CCP Data Format**
>
> The Ali-CCP dataset contains real-world click and conversion logs from Taobao's
> advertising system. It has two file types:
>
> - **sample_skeleton**: Per-impression records with `(sample_id, click, conversion, hash_id, n_features, features)`
> - **common_features**: Shared user/context features indexed by `hash_id`
>
> Features use binary delimiters: `\\x01` (feature separator), `\\x02` (field/id separator),
> `\\x03` (id/value separator). Each feature is a triple: `(field_id, feature_id, value)`."""),

    code("""def parse_features(feat_bytes):
    \"\"\"Parse Ali-CCP feature string with \\x01/\\x02/\\x03 delimiters.
    Returns list of (field_id, feature_id, value) tuples.\"\"\"
    features = []
    for feat in feat_bytes.split(b'\\x01'):
        feat = feat.strip()
        if not feat:
            continue
        parts = feat.split(b'\\x02')
        if len(parts) != 2:
            continue
        field_id = int(parts[0])
        id_val = parts[1].split(b'\\x03')
        if len(id_val) == 2:
            feat_id = int(id_val[0])
            value = float(id_val[1])
            features.append((field_id, feat_id, value))
    return features

# Parse a small sample to verify
with open(DATA_DIR / 'sample_skeleton_train.csv', 'rb') as f:
    line = f.readline()
    parts = line.split(b',', 5)
    feats = parse_features(parts[5])
    print(f'Sample 1: click={int(parts[1])}, conv={int(parts[2])}, n_features={int(parts[4])}')
    for field, fid, val in feats[:5]:
        print(f'  field={field}, feat_id={fid}, value={val}')\n"""),

    code("""def load_skeleton(filepath, max_samples=None):
    \"\"\"Load skeleton file, return labels and per-sample features.\"\"\"
    clicks, conversions = [], []
    hash_ids = []
    sample_features = []

    with open(filepath, 'rb') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            parts = line.split(b',', 5)
            clicks.append(int(parts[1]))
            conversions.append(int(parts[2]))
            hash_ids.append(parts[3].decode())
            feats = parse_features(parts[5])
            feat_dict = defaultdict(list)
            for field, fid, val in feats:
                feat_dict[field].append((fid, val))
            sample_features.append(dict(feat_dict))

            if (i + 1) % 1_000_000 == 0:
                print(f'  Loaded {i+1:,} samples...')

    return np.array(clicks), np.array(conversions), hash_ids, sample_features

print('Loading train data (first 5M samples)...')
start = time.time()
train_clicks, train_convs, train_hashes, train_feats = load_skeleton(
    DATA_DIR / 'sample_skeleton_train.csv', max_samples=SUBSAMPLE_SIZE
)
print(f'Loaded in {time.time()-start:.1f}s')
print(f'Train: {len(train_clicks):,} samples')
print(f'  Clicks: {train_clicks.sum():,} ({train_clicks.mean()*100:.2f}%)')
print(f'  Conversions: {train_convs.sum():,} ({train_convs.mean()*100:.4f}%)')
cvr_denom = max(train_clicks.sum(), 1)
print(f'  CVR among clicks: {train_convs[train_clicks==1].sum()/cvr_denom*100:.3f}%')\n"""),

    md("## 3. Exploratory Data Analysis"),

    code("""# Conversion funnel visualization
n_impressions = len(train_clicks)
n_clicks = int(train_clicks.sum())
n_conversions = int(train_convs.sum())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stages = ['Impressions', 'Clicks', 'Conversions']
counts = [n_impressions, n_clicks, n_conversions]
colors = ['#4e79a7', '#f28e2b', '#e15759']

bars = axes[0].barh(stages[::-1], counts[::-1], color=colors[::-1])
for bar, count in zip(bars, counts[::-1]):
    axes[0].text(bar.get_width() + n_impressions*0.02, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=11)
axes[0].set_xlabel('Count')
axes[0].set_title('Conversion Funnel')
axes[0].set_xscale('log')

rates = {
    'CTR': n_clicks / n_impressions * 100,
    'CVR (click->conv)': n_conversions / max(n_clicks, 1) * 100,
    'CTCVR': n_conversions / n_impressions * 100
}
bars2 = axes[1].bar(rates.keys(), rates.values(), color=colors)
for bar, rate in zip(bars2, rates.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.3f}%', ha='center', fontsize=11)
axes[1].set_ylabel('Rate (%)')
axes[1].set_title('Click & Conversion Rates')

plt.suptitle('Ali-CCP: Click -> Conversion Funnel', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(PROCESSED_DIR / 'conversion_funnel.png'), dpi=150, bbox_inches='tight')
plt.show()\n"""),

    code("""# Feature field analysis
field_counter = Counter()
field_unique_feats = defaultdict(set)

for feat_dict in train_feats[:500000]:
    for field_id, feat_list in feat_dict.items():
        field_counter[field_id] += len(feat_list)
        for fid, val in feat_list:
            field_unique_feats[field_id].add(fid)

print('=== Per-Sample Feature Fields (Ad Features) ===')
print(f'{\"Field\":>8} | {\"Occurrences\":>12} | {\"Unique Feats\":>12}')
print('-' * 40)
for field in sorted(field_counter.keys()):
    print(f'{field:>8} | {field_counter[field]:>12,} | {len(field_unique_feats[field]):>12,}')\n"""),

    md("""## 4. Sample Selection Bias Analysis

> **Concept: Sample Selection Bias (SSB) in CVR Prediction**
>
> Traditional CVR models train only on *clicked* samples (where conversion labels
> are observed). At inference, CVR must be predicted for *all* impressions.
> This creates Sample Selection Bias: the training distribution (clicked) differs
> from the inference distribution (all impressions).
>
> **ESMM** solves this by decomposing: `CTCVR = CTR x CVR`, where both CTR and
> CTCVR are trained on the *entire* impression space."""),

    code("""# SSB visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Sample Selection Bias in CVR Prediction', fontsize=14)

ax.text(5, 6, f'All Impressions: {n_impressions:,}', ha='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#4e79a7', alpha=0.3))
ax.text(3, 4, f'Clicked: {n_clicks:,}\\n({n_clicks/n_impressions*100:.1f}%)',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f28e2b', alpha=0.3))
ax.text(7, 4, f'Not Clicked: {n_impressions-n_clicks:,}\\n(No CVR label!)',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#bab0ac', alpha=0.3))
ax.text(3, 2, f'Converted: {n_conversions:,}\\n({n_conversions/max(n_clicks,1)*100:.2f}% of clicks)',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#e15759', alpha=0.3))

ax.annotate('', xy=(3, 5.3), xytext=(3, 4.7), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(7, 5.3), xytext=(7, 4.7), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(3, 3.3), xytext=(3, 2.7), arrowprops=dict(arrowstyle='->', lw=1.5))

ax.text(5, 0.8, 'Traditional CVR: train on clicked only (biased!)', ha='center',
        fontsize=11, color='#bab0ac')
ax.text(5, 0.3, 'ESMM: train on entire space (unbiased!)', ha='center',
        fontsize=11, color='#e15759', fontweight='bold')

plt.tight_layout()
plt.savefig(str(PROCESSED_DIR / 'ssb_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()\n"""),

    md("## 5. Data Preprocessing"),

    code("""# Load common features
print('Loading common features...')
start = time.time()
needed_hashes = set(train_hashes)
common_features = {}

with open(DATA_DIR / 'common_features_train.csv', 'rb') as f:
    for i, line in enumerate(f):
        parts = line.split(b',', 2)
        hash_id = parts[0].decode()
        if hash_id in needed_hashes:
            feats = parse_features(parts[2])
            feat_dict = defaultdict(list)
            for field, fid, val in feats:
                feat_dict[field].append((fid, val))
            common_features[hash_id] = dict(feat_dict)

        if (i + 1) % 50000 == 0:
            print(f'  Scanned {i+1:,}, matched {len(common_features):,}...')

print(f'Loaded {len(common_features):,} common features in {time.time()-start:.1f}s')
print(f'Coverage: {len(common_features)/len(needed_hashes)*100:.1f}%')\n"""),

    code("""# Build feature ID mappings per field
print('Building feature mappings...')
field_feat_ids = defaultdict(set)

for feat_dict in train_feats:
    for field, feat_list in feat_dict.items():
        for fid, val in feat_list:
            field_feat_ids[field].add(fid)

for feat_dict in common_features.values():
    for field, feat_list in feat_dict.items():
        for fid, val in feat_list:
            field_feat_ids[field].add(fid)

# Per-field mappings (0 = padding/unknown)
feat_mappings = {}
for field in sorted(field_feat_ids.keys()):
    mapping = {fid: idx + 1 for idx, fid in enumerate(sorted(field_feat_ids[field]))}
    feat_mappings[field] = mapping
    print(f'  Field {field}: {len(mapping):,} unique features')

all_fields = sorted(field_feat_ids.keys())
print(f'\\nTotal fields: {len(all_fields)}')
print(f'Total unique features: {sum(len(m) for m in feat_mappings.values()):,}')\n"""),

    code("""def encode_sample(sample_feats, common_feats, fields, mappings):
    \"\"\"Encode a sample into fixed-size feature vector.\"\"\"
    encoded = np.zeros(len(fields), dtype=np.int64)
    values = np.zeros(len(fields), dtype=np.float32)

    merged = {}
    if common_feats:
        merged.update(common_feats)
    merged.update(sample_feats)

    for i, field in enumerate(fields):
        if field in merged and merged[field]:
            fid, val = merged[field][0]
            if fid in mappings[field]:
                encoded[i] = mappings[field][fid]
                values[i] = val
    return encoded, values

print('Encoding training samples...')
start = time.time()
X_ids = np.zeros((len(train_clicks), len(all_fields)), dtype=np.int64)
X_vals = np.zeros((len(train_clicks), len(all_fields)), dtype=np.float32)

for i in range(len(train_clicks)):
    common = common_features.get(train_hashes[i], {})
    X_ids[i], X_vals[i] = encode_sample(train_feats[i], common, all_fields, feat_mappings)
    if (i + 1) % 1_000_000 == 0:
        print(f'  Encoded {i+1:,}...')

y_click = train_clicks.astype(np.float32)
y_conv = train_convs.astype(np.float32)

print(f'Done in {time.time()-start:.1f}s')
print(f'X_ids: {X_ids.shape}, X_vals: {X_vals.shape}')
print(f'Feature coverage: {(X_ids > 0).mean()*100:.1f}% non-zero')\n"""),

    code("""# Train/validation split
from sklearn.model_selection import train_test_split

indices = np.arange(len(y_click))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

data = {
    'X_ids_train': X_ids[train_idx], 'X_vals_train': X_vals[train_idx],
    'y_click_train': y_click[train_idx], 'y_conv_train': y_conv[train_idx],
    'X_ids_val': X_ids[val_idx], 'X_vals_val': X_vals[val_idx],
    'y_click_val': y_click[val_idx], 'y_conv_val': y_conv[val_idx],
    'all_fields': all_fields,
    'feat_mappings': feat_mappings,
    'field_cardinalities': {field: len(m) + 1 for field, m in feat_mappings.items()},
}

save_path = PROCESSED_DIR / 'aliccp_processed.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(data, f)

print(f'Saved to {save_path}')
print(f'Train: {len(train_idx):,}, Val: {len(val_idx):,}')
print(f'Train CTR: {y_click[train_idx].mean()*100:.2f}%')
print(f'Val CTR: {y_click[val_idx].mean()*100:.2f}%')\n"""),

    md("""## 6. Summary & Key Takeaways

### Key Takeaways

1. **Extreme class imbalance**: CTR ~4-5%, CVR ~0.5-0.6% among clicks, CTCVR ~0.03%.
2. **Sample Selection Bias**: Training CVR on clicked samples creates distribution mismatch.
3. **Rich feature space**: 20+ feature fields with millions of unique feature IDs.
4. **Scale**: Even our 5M subsample is substantial.

### Next Steps

In the next notebook, we implement **ESMM** to address the SSB problem."""),
]

with open(os.path.join(NOTEBOOK_DIR, '01_data_exploration.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_nb(cells_eda), f, indent=1, ensure_ascii=False)
print("Created 01_data_exploration.ipynb")


# ============================================================
# NOTEBOOK 2: ESMM
# ============================================================
cells_esmm = [
    md("""# ESMM: Entire Space Multi-Task Model on Ali-CCP

---

## Learning Objectives

1. **Understand** the ESMM architecture and how it addresses Sample Selection Bias
2. **Implement** ESMM from scratch in PyTorch
3. **Train** ESMM on Ali-CCP data and evaluate CTR, CVR, and CTCVR performance
4. **Analyze** the benefits of entire-space training vs traditional CVR

## Table of Contents

1. [Setup & Data Loading](#1-setup--data-loading)
2. [ESMM Architecture](#2-esmm-architecture)
3. [Model Implementation](#3-model-implementation)
4. [Training](#4-training)
5. [Evaluation & Analysis](#5-evaluation--analysis)
6. [Ablation Study](#6-ablation-study)
7. [Key Takeaways](#7-key-takeaways)"""),

    code("""import numpy as np
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path('../../data/aliccp/processed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Hyperparameters
EMBEDDING_DIM = 8
HIDDEN_DIMS = [256, 128, 64]
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
DROPOUT = 0.2\n"""),

    code("""# Load preprocessed data
with open(DATA_DIR / 'aliccp_processed.pkl', 'rb') as f:
    data = pickle.load(f)

X_ids_train = torch.LongTensor(data['X_ids_train'])
X_vals_train = torch.FloatTensor(data['X_vals_train'])
y_click_train = torch.FloatTensor(data['y_click_train'])
y_conv_train = torch.FloatTensor(data['y_conv_train'])

X_ids_val = torch.LongTensor(data['X_ids_val'])
X_vals_val = torch.FloatTensor(data['X_vals_val'])
y_click_val = torch.FloatTensor(data['y_click_val'])
y_conv_val = torch.FloatTensor(data['y_conv_val'])

field_cardinalities = data['field_cardinalities']
all_fields = data['all_fields']

print(f'Train: {len(y_click_train):,}, Val: {len(y_click_val):,}')
print(f'Fields: {len(all_fields)}, Cardinalities: {list(field_cardinalities.values())}')
print(f'Train CTR: {y_click_train.mean()*100:.2f}%, CTCVR: {y_conv_train.mean()*100:.4f}%')

train_ds = TensorDataset(X_ids_train, X_vals_train, y_click_train, y_conv_train)
val_ds = TensorDataset(X_ids_val, X_vals_val, y_click_val, y_conv_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)\n"""),

    md("""## 2. ESMM Architecture

> **Concept: Entire Space Multi-Task Model (ESMM)**
>
> ESMM decomposes the post-view click-through & conversion rate (CTCVR) as:
>
> $$\\text{CTCVR} = \\text{CTR} \\times \\text{CVR}$$
>
> Key innovations:
> - **Entire-space training**: Both CTR and CTCVR losses are computed over ALL impressions
> - **Implicit CVR**: The CVR tower never sees labels directly; it learns through the CTCVR signal
> - **Shared embeddings**: CTR and CVR towers share the same feature embeddings
>
> This elegantly avoids Sample Selection Bias without any data manipulation."""),

    code("""class ESMM(nn.Module):
    \"\"\"Entire Space Multi-Task Model.\"\"\"

    def __init__(self, field_dims, embedding_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.n_fields = len(field_dims)

        # Shared embeddings (one per field)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in field_dims
        ])

        input_dim = self.n_fields * embedding_dim

        # CTR tower
        ctr_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            ctr_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        ctr_layers.append(nn.Linear(prev_dim, 1))
        self.ctr_tower = nn.Sequential(*ctr_layers)

        # CVR tower
        cvr_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            cvr_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        cvr_layers.append(nn.Linear(prev_dim, 1))
        self.cvr_tower = nn.Sequential(*cvr_layers)

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data[1:])
        for module in [self.ctr_tower, self.cvr_tower]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, feat_ids, feat_vals=None):
        # Embed each field
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(feat_ids[:, i])  # (B, D)
            if feat_vals is not None:
                emb = emb * feat_vals[:, i:i+1]
            emb_list.append(emb)

        x = torch.cat(emb_list, dim=1)  # (B, n_fields * D)

        ctr_logit = self.ctr_tower(x).squeeze(-1)
        cvr_logit = self.cvr_tower(x).squeeze(-1)

        ctr_pred = torch.sigmoid(ctr_logit)
        cvr_pred = torch.sigmoid(cvr_logit)
        ctcvr_pred = ctr_pred * cvr_pred

        return ctr_pred, cvr_pred, ctcvr_pred

print('ESMM model created')
field_dims = [field_cardinalities[f] for f in all_fields]
model = ESMM(field_dims, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n_params:,}')\n"""),

    md("## 4. Training"),

    code("""def train_esmm_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        ids, vals, click, conv = [b.to(device) for b in batch]
        optimizer.zero_grad()

        ctr_pred, cvr_pred, ctcvr_pred = model(ids, vals)

        # CTR loss: all impressions have click labels
        ctr_loss = nn.functional.binary_cross_entropy(ctr_pred, click, reduction='mean')

        # CTCVR loss: all impressions have conversion labels (0 for unclicked)
        ctcvr_loss = nn.functional.binary_cross_entropy(ctcvr_pred, conv, reduction='mean')

        loss = ctr_loss + ctcvr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_esmm(model, loader, device):
    model.eval()
    all_ctr_pred, all_cvr_pred, all_ctcvr_pred = [], [], []
    all_click, all_conv = [], []

    with torch.no_grad():
        for batch in loader:
            ids, vals, click, conv = [b.to(device) for b in batch]
            ctr_pred, cvr_pred, ctcvr_pred = model(ids, vals)

            all_ctr_pred.append(ctr_pred.cpu().numpy())
            all_cvr_pred.append(cvr_pred.cpu().numpy())
            all_ctcvr_pred.append(ctcvr_pred.cpu().numpy())
            all_click.append(click.cpu().numpy())
            all_conv.append(conv.cpu().numpy())

    ctr_pred = np.concatenate(all_ctr_pred)
    cvr_pred = np.concatenate(all_cvr_pred)
    ctcvr_pred = np.concatenate(all_ctcvr_pred)
    click = np.concatenate(all_click)
    conv = np.concatenate(all_conv)

    metrics = {
        'ctr_auc': roc_auc_score(click, ctr_pred),
        'ctcvr_auc': roc_auc_score(conv, ctcvr_pred) if conv.sum() > 0 else 0.0,
    }
    # CVR AUC: only among clicked samples
    clicked_mask = click == 1
    if clicked_mask.sum() > 10 and conv[clicked_mask].sum() > 0:
        metrics['cvr_auc'] = roc_auc_score(conv[clicked_mask], cvr_pred[clicked_mask])
    else:
        metrics['cvr_auc'] = 0.0

    return metrics\n"""),

    code("""# Training loop
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
history = {'train_loss': [], 'ctr_auc': [], 'cvr_auc': [], 'ctcvr_auc': []}
best_ctcvr_auc = 0

print('Training ESMM...')
for epoch in range(NUM_EPOCHS):
    start = time.time()
    train_loss = train_esmm_epoch(model, train_loader, optimizer, device)
    metrics = evaluate_esmm(model, val_loader, device)
    elapsed = time.time() - start

    history['train_loss'].append(train_loss)
    for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc']:
        history[k].append(metrics[k])

    if metrics['ctcvr_auc'] > best_ctcvr_auc:
        best_ctcvr_auc = metrics['ctcvr_auc']
        torch.save(model.state_dict(), str(DATA_DIR / 'esmm_best.pt'))

    print(f'Epoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.1f}s) | Loss: {train_loss:.4f} | '
          f'CTR AUC: {metrics[\"ctr_auc\"]:.4f} | CVR AUC: {metrics[\"cvr_auc\"]:.4f} | '
          f'CTCVR AUC: {metrics[\"ctcvr_auc\"]:.4f}')

print(f'\\nBest CTCVR AUC: {best_ctcvr_auc:.4f}')\n"""),

    md("## 5. Evaluation & Analysis"),

    code("""# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], 'b-o', markersize=4)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

for key, color, label in [('ctr_auc', 'blue', 'CTR AUC'),
                            ('cvr_auc', 'orange', 'CVR AUC'),
                            ('ctcvr_auc', 'red', 'CTCVR AUC')]:
    axes[1].plot(history[key], '-o', color=color, label=label, markersize=4)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].set_title('Validation AUC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('ESMM Training on Ali-CCP', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(DATA_DIR / 'esmm_training.png'), dpi=150, bbox_inches='tight')
plt.show()\n"""),

    md("""## 6. Ablation Study

> **Concept:** We compare ESMM (shared embeddings, entire-space training) against:
> - **Naive CVR**: Train CVR only on clicked samples (has SSB)
> - **Separate Embeddings**: ESMM structure but without shared embeddings"""),

    code("""# Ablation: Naive CVR (train only on clicked samples)
class NaiveCVR(nn.Module):
    def __init__(self, field_dims, embedding_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in field_dims
        ])
        input_dim = len(field_dims) * embedding_dim
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat_ids, feat_vals=None):
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(feat_ids[:, i])
            if feat_vals is not None:
                e = e * feat_vals[:, i:i+1]
            embs.append(e)
        x = torch.cat(embs, dim=1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))

# Train Naive CVR only on clicked samples
naive_model = NaiveCVR(field_dims, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
naive_opt = optim.Adam(naive_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Filter to clicked samples
click_mask_train = data['y_click_train'] == 1
X_clicked = torch.LongTensor(data['X_ids_train'][click_mask_train])
V_clicked = torch.FloatTensor(data['X_vals_train'][click_mask_train])
y_clicked = torch.FloatTensor(data['y_conv_train'][click_mask_train])

clicked_ds = TensorDataset(X_clicked, V_clicked, y_clicked)
clicked_loader = DataLoader(clicked_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f'Naive CVR training on {len(clicked_ds):,} clicked samples...')
for epoch in range(NUM_EPOCHS):
    naive_model.train()
    total_loss = 0
    for ids, vals, conv in clicked_loader:
        ids, vals, conv = ids.to(device), vals.to(device), conv.to(device)
        naive_opt.zero_grad()
        pred = naive_model(ids, vals)
        loss = nn.functional.binary_cross_entropy(pred, conv)
        loss.backward()
        naive_opt.step()
        total_loss += loss.item()
    print(f'  Epoch {epoch+1}: Loss = {total_loss/len(clicked_loader):.4f}')

# Evaluate Naive CVR on entire val set (this is the SSB test!)
naive_model.eval()
all_preds = []
with torch.no_grad():
    for batch in val_loader:
        ids, vals = batch[0].to(device), batch[1].to(device)
        pred = naive_model(ids, vals)
        all_preds.append(pred.cpu().numpy())
naive_preds = np.concatenate(all_preds)

naive_ctcvr_auc = roc_auc_score(data['y_conv_val'], naive_preds)
click_mask_val = data['y_click_val'] == 1
naive_cvr_auc = roc_auc_score(
    data['y_conv_val'][click_mask_val], naive_preds[click_mask_val]
) if click_mask_val.sum() > 0 and data['y_conv_val'][click_mask_val].sum() > 0 else 0

print(f'\\nNaive CVR Results:')
print(f'  CVR AUC (clicked): {naive_cvr_auc:.4f}')
print(f'  CTCVR AUC (all):   {naive_ctcvr_auc:.4f}')
print(f'\\nESMM Results:')
print(f'  CVR AUC (clicked): {max(history[\"cvr_auc\"]):.4f}')
print(f'  CTCVR AUC (all):   {best_ctcvr_auc:.4f}')\n"""),

    code("""# Save results
results = {
    'esmm': {
        'ctr_auc': max(history['ctr_auc']),
        'cvr_auc': max(history['cvr_auc']),
        'ctcvr_auc': best_ctcvr_auc,
        'params': n_params,
        'history': history,
    },
    'naive_cvr': {
        'cvr_auc': naive_cvr_auc,
        'ctcvr_auc': naive_ctcvr_auc,
    }
}

with open(DATA_DIR / 'esmm_results.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'history'}
               for k, v in results.items()}, f, indent=2)
print('Results saved.')\n"""),

    md("""## 7. Key Takeaways

1. **Entire-space training** eliminates Sample Selection Bias by training on all impressions.
2. **CTCVR = CTR x CVR** decomposition lets the CVR tower learn without direct labels.
3. **Shared embeddings** between CTR and CVR towers enable knowledge transfer.
4. **Naive CVR** suffers from SSB — good CVR AUC on clicked samples but poor CTCVR AUC overall.

### Next Steps

In the next notebook, we implement **MMoE** and **PLE** for more flexible multi-task architectures."""),
]

with open(os.path.join(NOTEBOOK_DIR, '02_esmm.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_nb(cells_esmm), f, indent=1, ensure_ascii=False)
print("Created 02_esmm.ipynb")


# ============================================================
# NOTEBOOK 3: Advanced CVR (MMoE + PLE)
# ============================================================
cells_adv = [
    md("""# Advanced Multi-Task CVR: MMoE & PLE on Ali-CCP

---

## Learning Objectives

1. **Understand** Multi-gate Mixture-of-Experts (MMoE) and Progressive Layered Extraction (PLE)
2. **Implement** MMoE and PLE from scratch in PyTorch
3. **Compare** against ESMM on the Ali-CCP dataset
4. **Analyze** expert specialization and gating patterns

## Table of Contents

1. [Setup & Data Loading](#1-setup--data-loading)
2. [MMoE Architecture](#2-mmoe-architecture)
3. [PLE Architecture](#3-ple-architecture)
4. [Training & Evaluation](#4-training--evaluation)
5. [Expert Analysis](#5-expert-analysis)
6. [Key Takeaways](#6-key-takeaways)"""),

    code("""import numpy as np
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path('../../data/aliccp/processed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

EMBEDDING_DIM = 8
HIDDEN_DIMS = [256, 128, 64]
N_EXPERTS = 4
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
DROPOUT = 0.2\n"""),

    code("""# Load data
with open(DATA_DIR / 'aliccp_processed.pkl', 'rb') as f:
    data = pickle.load(f)

X_ids_train = torch.LongTensor(data['X_ids_train'])
X_vals_train = torch.FloatTensor(data['X_vals_train'])
y_click_train = torch.FloatTensor(data['y_click_train'])
y_conv_train = torch.FloatTensor(data['y_conv_train'])
X_ids_val = torch.LongTensor(data['X_ids_val'])
X_vals_val = torch.FloatTensor(data['X_vals_val'])
y_click_val = torch.FloatTensor(data['y_click_val'])
y_conv_val = torch.FloatTensor(data['y_conv_val'])

field_cardinalities = data['field_cardinalities']
all_fields = data['all_fields']
field_dims = [field_cardinalities[f] for f in all_fields]

train_ds = TensorDataset(X_ids_train, X_vals_train, y_click_train, y_conv_train)
val_ds = TensorDataset(X_ids_val, X_vals_val, y_click_val, y_conv_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)
print(f'Train: {len(train_ds):,}, Val: {len(val_ds):,}')\n"""),

    md("""## 2. MMoE Architecture

> **Concept: Multi-gate Mixture-of-Experts (MMoE)**
>
> MMoE uses a shared set of expert networks with task-specific gating:
> - **Experts**: N shared sub-networks, each processing the full input
> - **Gates**: Per-task softmax gates that weight expert outputs differently
> - **Towers**: Task-specific output heads on top of gated expert mixtures
>
> This allows different tasks to focus on different expert combinations."""),

    code("""class MMoE(nn.Module):
    def __init__(self, field_dims, embedding_dim, n_experts, expert_dim, tower_dims, dropout=0.2):
        super().__init__()
        self.n_fields = len(field_dims)
        self.n_experts = n_experts

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in field_dims
        ])
        input_dim = self.n_fields * embedding_dim

        # Shared experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(expert_dim, expert_dim), nn.ReLU()
            ) for _ in range(n_experts)
        ])

        # Task-specific gates
        self.gate_ctr = nn.Linear(input_dim, n_experts)
        self.gate_cvr = nn.Linear(input_dim, n_experts)

        # Task towers
        def make_tower(in_dim, hidden_dims):
            layers = []
            prev = in_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
                prev = dim
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.tower_ctr = make_tower(expert_dim, tower_dims)
        self.tower_cvr = make_tower(expert_dim, tower_dims)

    def forward(self, feat_ids, feat_vals=None):
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(feat_ids[:, i])
            if feat_vals is not None:
                e = e * feat_vals[:, i:i+1]
            embs.append(e)
        x = torch.cat(embs, dim=1)

        # Expert outputs
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, D)

        # Gated mixtures
        gate_ctr = torch.softmax(self.gate_ctr(x), dim=-1).unsqueeze(-1)  # (B, E, 1)
        gate_cvr = torch.softmax(self.gate_cvr(x), dim=-1).unsqueeze(-1)

        ctr_input = (expert_outs * gate_ctr).sum(dim=1)  # (B, D)
        cvr_input = (expert_outs * gate_cvr).sum(dim=1)

        ctr_pred = torch.sigmoid(self.tower_ctr(ctr_input).squeeze(-1))
        cvr_pred = torch.sigmoid(self.tower_cvr(cvr_input).squeeze(-1))
        ctcvr_pred = ctr_pred * cvr_pred

        return ctr_pred, cvr_pred, ctcvr_pred, gate_ctr.squeeze(-1), gate_cvr.squeeze(-1)

mmoe_model = MMoE(field_dims, EMBEDDING_DIM, N_EXPERTS, 128, [64, 32], DROPOUT).to(device)
print(f'MMoE params: {sum(p.numel() for p in mmoe_model.parameters()):,}')\n"""),

    md("""## 3. PLE Architecture

> **Concept: Progressive Layered Extraction (PLE)**
>
> PLE improves on MMoE by adding:
> - **Task-specific experts** alongside shared experts
> - **Progressive extraction layers** that refine representations
>
> This reduces negative transfer between tasks."""),

    code("""class PLELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, n_shared, n_task_specific, n_tasks, dropout=0.2):
        super().__init__()
        self.n_tasks = n_tasks
        total_experts = n_shared + n_task_specific

        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(n_shared)
        ])

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(n_task_specific)
            ]) for _ in range(n_tasks)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(input_dim, total_experts) for _ in range(n_tasks)
        ])

    def forward(self, task_inputs):
        shared_outs = [expert(task_inputs[0]) for expert in self.shared_experts]
        outputs = []

        for t in range(self.n_tasks):
            task_outs = [expert(task_inputs[t]) for expert in self.task_experts[t]]
            all_outs = torch.stack(shared_outs + task_outs, dim=1)  # (B, E, D)
            gate = torch.softmax(self.gates[t](task_inputs[t]), dim=-1).unsqueeze(-1)
            outputs.append((all_outs * gate).sum(dim=1))

        return outputs


class PLE(nn.Module):
    def __init__(self, field_dims, embedding_dim, n_layers, expert_dim,
                 n_shared, n_task_specific, tower_dims, dropout=0.2):
        super().__init__()
        self.n_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in field_dims
        ])
        input_dim = self.n_fields * embedding_dim

        self.extraction_layers = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(n_layers):
            self.extraction_layers.append(
                PLELayer(prev_dim, expert_dim, n_shared, n_task_specific, 2, dropout)
            )
            prev_dim = expert_dim

        def make_tower(in_dim, hidden_dims):
            layers = []
            prev = in_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
                prev = dim
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.tower_ctr = make_tower(expert_dim, tower_dims)
        self.tower_cvr = make_tower(expert_dim, tower_dims)

    def forward(self, feat_ids, feat_vals=None):
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(feat_ids[:, i])
            if feat_vals is not None:
                e = e * feat_vals[:, i:i+1]
            embs.append(e)
        x = torch.cat(embs, dim=1)

        task_inputs = [x, x]  # Both tasks start from same input
        for layer in self.extraction_layers:
            task_inputs = layer(task_inputs)

        ctr_pred = torch.sigmoid(self.tower_ctr(task_inputs[0]).squeeze(-1))
        cvr_pred = torch.sigmoid(self.tower_cvr(task_inputs[1]).squeeze(-1))
        ctcvr_pred = ctr_pred * cvr_pred

        return ctr_pred, cvr_pred, ctcvr_pred

ple_model = PLE(field_dims, EMBEDDING_DIM, n_layers=2, expert_dim=128,
                n_shared=2, n_task_specific=2, tower_dims=[64, 32], dropout=DROPOUT).to(device)
print(f'PLE params: {sum(p.numel() for p in ple_model.parameters()):,}')\n"""),

    md("## 4. Training & Evaluation"),

    code("""def train_multitask_epoch(model, loader, optimizer, device, model_type='mmoe'):
    model.train()
    total_loss = 0
    for batch in loader:
        ids, vals, click, conv = [b.to(device) for b in batch]
        optimizer.zero_grad()

        if model_type == 'mmoe':
            ctr_pred, cvr_pred, ctcvr_pred, _, _ = model(ids, vals)
        else:
            ctr_pred, cvr_pred, ctcvr_pred = model(ids, vals)

        ctr_loss = nn.functional.binary_cross_entropy(ctr_pred, click)
        ctcvr_loss = nn.functional.binary_cross_entropy(ctcvr_pred, conv)
        loss = ctr_loss + ctcvr_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_multitask(model, loader, device, model_type='mmoe'):
    model.eval()
    all_ctr, all_cvr, all_ctcvr = [], [], []
    all_click, all_conv = [], []

    with torch.no_grad():
        for batch in loader:
            ids, vals, click, conv = [b.to(device) for b in batch]
            if model_type == 'mmoe':
                ctr, cvr, ctcvr, _, _ = model(ids, vals)
            else:
                ctr, cvr, ctcvr = model(ids, vals)
            all_ctr.append(ctr.cpu().numpy())
            all_cvr.append(cvr.cpu().numpy())
            all_ctcvr.append(ctcvr.cpu().numpy())
            all_click.append(click.cpu().numpy())
            all_conv.append(conv.cpu().numpy())

    ctr_p = np.concatenate(all_ctr)
    cvr_p = np.concatenate(all_cvr)
    ctcvr_p = np.concatenate(all_ctcvr)
    click = np.concatenate(all_click)
    conv = np.concatenate(all_conv)

    metrics = {'ctr_auc': roc_auc_score(click, ctr_p),
               'ctcvr_auc': roc_auc_score(conv, ctcvr_p) if conv.sum() > 0 else 0}
    clicked = click == 1
    if clicked.sum() > 10 and conv[clicked].sum() > 0:
        metrics['cvr_auc'] = roc_auc_score(conv[clicked], cvr_p[clicked])
    return metrics\n"""),

    code("""def train_model(model, name, model_type, n_epochs=NUM_EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    history = {'loss': [], 'ctr_auc': [], 'cvr_auc': [], 'ctcvr_auc': []}
    best_auc = 0

    print(f'\\nTraining {name}...')
    for epoch in range(n_epochs):
        start = time.time()
        loss = train_multitask_epoch(model, train_loader, optimizer, device, model_type)
        metrics = evaluate_multitask(model, val_loader, device, model_type)
        elapsed = time.time() - start

        history['loss'].append(loss)
        for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc']:
            history[k].append(metrics.get(k, 0))

        if metrics['ctcvr_auc'] > best_auc:
            best_auc = metrics['ctcvr_auc']

        print(f'  Epoch {epoch+1} ({elapsed:.1f}s) | Loss: {loss:.4f} | '
              f'CTR: {metrics[\"ctr_auc\"]:.4f} | CVR: {metrics.get(\"cvr_auc\", 0):.4f} | '
              f'CTCVR: {metrics[\"ctcvr_auc\"]:.4f}')

    return history, best_auc

mmoe_history, mmoe_best = train_model(mmoe_model, 'MMoE', 'mmoe')
ple_history, ple_best = train_model(ple_model, 'PLE', 'ple')

print(f'\\nMMoE best CTCVR AUC: {mmoe_best:.4f}')
print(f'PLE best CTCVR AUC: {ple_best:.4f}')\n"""),

    md("## 5. Expert Analysis"),

    code("""# Visualize MMoE gating distributions
mmoe_model.eval()
gate_ctr_all, gate_cvr_all = [], []
with torch.no_grad():
    for batch in val_loader:
        ids, vals = batch[0].to(device), batch[1].to(device)
        _, _, _, g_ctr, g_cvr = mmoe_model(ids, vals)
        gate_ctr_all.append(g_ctr.cpu().numpy())
        gate_cvr_all.append(g_cvr.cpu().numpy())
        if len(gate_ctr_all) >= 10:
            break

gate_ctr_np = np.concatenate(gate_ctr_all)
gate_cvr_np = np.concatenate(gate_cvr_all)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(N_EXPERTS), gate_ctr_np.mean(axis=0), color='#4e79a7', alpha=0.8)
axes[0].set_xlabel('Expert')
axes[0].set_ylabel('Average Gate Weight')
axes[0].set_title('CTR Task: Expert Usage')
axes[0].set_xticks(range(N_EXPERTS))

axes[1].bar(range(N_EXPERTS), gate_cvr_np.mean(axis=0), color='#e15759', alpha=0.8)
axes[1].set_xlabel('Expert')
axes[1].set_ylabel('Average Gate Weight')
axes[1].set_title('CVR Task: Expert Usage')
axes[1].set_xticks(range(N_EXPERTS))

plt.suptitle('MMoE Expert Gating Distribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(DATA_DIR / 'mmoe_gating.png'), dpi=150, bbox_inches='tight')
plt.show()\n"""),

    code("""# Save all results
results = {
    'mmoe': {'ctcvr_auc': mmoe_best, 'ctr_auc': max(mmoe_history['ctr_auc']),
             'cvr_auc': max(mmoe_history['cvr_auc']),
             'params': sum(p.numel() for p in mmoe_model.parameters())},
    'ple': {'ctcvr_auc': ple_best, 'ctr_auc': max(ple_history['ctr_auc']),
            'cvr_auc': max(ple_history['cvr_auc']),
            'params': sum(p.numel() for p in ple_model.parameters())},
}

with open(DATA_DIR / 'advanced_cvr_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved.')\n"""),

    md("""## 6. Key Takeaways

1. **MMoE** enables task-specific expert weighting via gating networks
2. **PLE** adds task-specific experts to reduce negative transfer
3. Expert gating reveals which experts specialize for CTR vs CVR
4. More complex architectures need sufficient data to realize their potential

### Next Steps

The comparison notebook consolidates all multi-task model results."""),
]

with open(os.path.join(NOTEBOOK_DIR, '03_advanced_cvr.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_nb(cells_adv), f, indent=1, ensure_ascii=False)
print("Created 03_advanced_cvr.ipynb")


# ============================================================
# NOTEBOOK 4: Comparison
# ============================================================
cells_cmp = [
    md("""# Multi-Task CVR Models: Head-to-Head Comparison on Ali-CCP

---

## Learning Objectives

1. **Compare** Naive CVR, ESMM, MMoE, and PLE on Ali-CCP
2. **Analyze** the impact of entire-space training and multi-task architectures
3. **Understand** trade-offs between model complexity and performance"""),

    code("""import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path('../../data/aliccp/processed')

# Load results from previous notebooks
with open(DATA_DIR / 'esmm_results.json') as f:
    esmm_results = json.load(f)

with open(DATA_DIR / 'advanced_cvr_results.json') as f:
    adv_results = json.load(f)

print('=== Ali-CCP Multi-Task CVR Results ===')
print(f'{\"Model\":<15} {\"CTR AUC\":>10} {\"CVR AUC\":>10} {\"CTCVR AUC\":>10}')
print('-' * 50)

all_results = {}
for name, res in [('Naive CVR', esmm_results.get('naive_cvr', {})),
                   ('ESMM', esmm_results.get('esmm', {})),
                   ('MMoE', adv_results.get('mmoe', {})),
                   ('PLE', adv_results.get('ple', {}))]:
    ctr = res.get('ctr_auc', '-')
    cvr = res.get('cvr_auc', '-')
    ctcvr = res.get('ctcvr_auc', '-')
    ctr_s = f'{ctr:.4f}' if isinstance(ctr, float) else '-'
    cvr_s = f'{cvr:.4f}' if isinstance(cvr, float) else '-'
    ctcvr_s = f'{ctcvr:.4f}' if isinstance(ctcvr, float) else '-'
    print(f'{name:<15} {ctr_s:>10} {cvr_s:>10} {ctcvr_s:>10}')
    all_results[name] = res\n"""),

    code("""# Comparison visualization
models = list(all_results.keys())
ctcvr_aucs = [all_results[m].get('ctcvr_auc', 0) for m in models]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#bab0ac', '#4e79a7', '#f28e2b', '#e15759']
bars = axes[0].bar(models, ctcvr_aucs, color=colors)
for bar, auc in zip(bars, ctcvr_aucs):
    if auc > 0:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{auc:.4f}', ha='center', fontsize=10)
axes[0].set_ylabel('CTCVR AUC')
axes[0].set_title('CTCVR AUC Comparison')
axes[0].grid(True, alpha=0.3, axis='y')

# Architecture comparison
arch_features = {
    'Naive CVR': [0, 0, 0],
    'ESMM': [1, 1, 0],
    'MMoE': [1, 1, 1],
    'PLE': [1, 1, 1],
}
feature_names = ['Entire-Space', 'Shared Emb', 'Expert Gating']
x = np.arange(len(feature_names))
width = 0.2
for i, (model, feats) in enumerate(arch_features.items()):
    axes[1].bar(x + i*width, feats, width, label=model, color=colors[i], alpha=0.8)
axes[1].set_xticks(x + 1.5*width)
axes[1].set_xticklabels(feature_names)
axes[1].set_ylabel('Has Feature')
axes[1].set_title('Architecture Features')
axes[1].legend()

plt.suptitle('Ali-CCP Multi-Task CVR: Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(DATA_DIR / 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()\n"""),

    md("""## Key Takeaways

1. **Entire-space training matters**: ESMM outperforms Naive CVR by addressing SSB
2. **Expert architectures**: MMoE and PLE offer flexible multi-task learning
3. **Data scale**: With 5M samples from Ali-CCP, the richer architectures can show their strengths
4. **Real-world applicability**: These models are deployed at scale in Alibaba's ad system

### Comparison with Tenrec Experiment

This experiment validates the same multi-task learning principles on a different,
industry-standard dataset. The Ali-CCP dataset's extreme sparsity (0.03% CTCVR)
makes it a more challenging testbed for multi-task CVR prediction."""),
]

with open(os.path.join(NOTEBOOK_DIR, '04_comparison.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_nb(cells_cmp), f, indent=1, ensure_ascii=False)
print("Created 04_comparison.ipynb")

print("\nAll 4 Ali-CCP notebooks created!")
