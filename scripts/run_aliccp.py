"""Run complete Ali-CCP multi-task CVR experiment on GPU.
Covers: data preprocessing, ESMM, Naive CVR, MMoE, PLE.
"""
import numpy as np
import pickle
import time
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path('data/aliccp')
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================
# STEP 1: Data Preprocessing
# ============================================================
SUBSAMPLE_SIZE = 5_000_000
processed_path = PROCESSED_DIR / 'aliccp_processed.pkl'

def parse_features(feat_bytes):
    features = []
    for feat in feat_bytes.split(b'\x01'):
        feat = feat.strip()
        if not feat:
            continue
        parts = feat.split(b'\x02')
        if len(parts) != 2:
            continue
        field_id = int(parts[0])
        id_val = parts[1].split(b'\x03')
        if len(id_val) == 2:
            feat_id = int(id_val[0])
            value = float(id_val[1])
            features.append((field_id, feat_id, value))
    return features


if processed_path.exists():
    print(f'\nLoading preprocessed data from {processed_path}...')
    with open(processed_path, 'rb') as f:
        data = pickle.load(f)
    print('Loaded.')
else:
    print(f'\n{"="*60}')
    print('STEP 1: Data Preprocessing')
    print(f'{"="*60}')

    # Load skeleton
    print(f'Loading train skeleton ({SUBSAMPLE_SIZE/1e6:.0f}M samples)...')
    start = time.time()
    clicks, convs, hash_ids, sample_features = [], [], [], []

    with open(DATA_DIR / 'sample_skeleton_train.csv', 'rb') as f:
        for i, line in enumerate(f):
            if i >= SUBSAMPLE_SIZE:
                break
            parts = line.split(b',', 5)
            clicks.append(int(parts[1]))
            convs.append(int(parts[2]))
            hash_ids.append(parts[3].decode())
            feats = parse_features(parts[5])
            feat_dict = defaultdict(list)
            for field, fid, val in feats:
                feat_dict[field].append((fid, val))
            sample_features.append(dict(feat_dict))
            if (i + 1) % 1_000_000 == 0:
                print(f'  {i+1:,} samples loaded...')

    clicks = np.array(clicks)
    convs = np.array(convs)
    print(f'Loaded {len(clicks):,} samples in {time.time()-start:.1f}s')
    print(f'  CTR: {clicks.mean()*100:.2f}%, CTCVR: {convs.mean()*100:.4f}%')

    # Load common features
    print('Loading common features...')
    start = time.time()
    needed = set(hash_ids)
    common_features = {}
    with open(DATA_DIR / 'common_features_train.csv', 'rb') as f:
        for i, line in enumerate(f):
            parts = line.split(b',', 2)
            h = parts[0].decode()
            if h in needed:
                feats = parse_features(parts[2])
                fd = defaultdict(list)
                for field, fid, val in feats:
                    fd[field].append((fid, val))
                common_features[h] = dict(fd)
            if (i + 1) % 50000 == 0:
                print(f'  Scanned {i+1:,}, matched {len(common_features):,}')
    print(f'Loaded {len(common_features):,} common features in {time.time()-start:.1f}s')

    # Build feature mappings
    print('Building feature mappings...')
    field_feat_ids = defaultdict(set)
    for fd in sample_features:
        for field, fl in fd.items():
            for fid, _ in fl:
                field_feat_ids[field].add(fid)
    for fd in common_features.values():
        for field, fl in fd.items():
            for fid, _ in fl:
                field_feat_ids[field].add(fid)

    feat_mappings = {}
    for field in sorted(field_feat_ids.keys()):
        mapping = {fid: idx + 1 for idx, fid in enumerate(sorted(field_feat_ids[field]))}
        feat_mappings[field] = mapping
        print(f'  Field {field}: {len(mapping):,} features')

    all_fields = sorted(field_feat_ids.keys())

    # Encode samples
    print('Encoding samples...')
    start = time.time()
    X_ids = np.zeros((len(clicks), len(all_fields)), dtype=np.int64)
    X_vals = np.zeros((len(clicks), len(all_fields)), dtype=np.float32)

    for i in range(len(clicks)):
        merged = {}
        c = common_features.get(hash_ids[i], {})
        if c:
            merged.update(c)
        merged.update(sample_features[i])
        for j, field in enumerate(all_fields):
            if field in merged and merged[field]:
                fid, val = merged[field][0]
                if fid in feat_mappings[field]:
                    X_ids[i, j] = feat_mappings[field][fid]
                    X_vals[i, j] = val
        if (i + 1) % 1_000_000 == 0:
            print(f'  Encoded {i+1:,}...')

    print(f'Encoded in {time.time()-start:.1f}s')

    # Split
    indices = np.arange(len(clicks))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    data = {
        'X_ids_train': X_ids[train_idx], 'X_vals_train': X_vals[train_idx],
        'y_click_train': clicks[train_idx].astype(np.float32),
        'y_conv_train': convs[train_idx].astype(np.float32),
        'X_ids_val': X_ids[val_idx], 'X_vals_val': X_vals[val_idx],
        'y_click_val': clicks[val_idx].astype(np.float32),
        'y_conv_val': convs[val_idx].astype(np.float32),
        'all_fields': all_fields,
        'feat_mappings': feat_mappings,
        'field_cardinalities': {f: len(m) + 1 for f, m in feat_mappings.items()},
    }

    with open(processed_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {processed_path}')

# ============================================================
# STEP 2: Setup data loaders
# ============================================================
print(f'\n{"="*60}')
print('STEP 2: Setup')
print(f'{"="*60}')

EMBEDDING_DIM = 8
HIDDEN_DIMS = [256, 128, 64]
BATCH_SIZE = 4096
LR = 1e-3
NUM_EPOCHS = 5
DROPOUT = 0.2
N_EXPERTS = 4

field_cardinalities = data['field_cardinalities']
all_fields = data['all_fields']
field_dims = [field_cardinalities[f] for f in all_fields]

X_ids_train = torch.LongTensor(data['X_ids_train'])
X_vals_train = torch.FloatTensor(data['X_vals_train'])
y_click_train = torch.FloatTensor(data['y_click_train'])
y_conv_train = torch.FloatTensor(data['y_conv_train'])
X_ids_val = torch.LongTensor(data['X_ids_val'])
X_vals_val = torch.FloatTensor(data['X_vals_val'])
y_click_val = torch.FloatTensor(data['y_click_val'])
y_conv_val = torch.FloatTensor(data['y_conv_val'])

train_ds = TensorDataset(X_ids_train, X_vals_train, y_click_train, y_conv_train)
val_ds = TensorDataset(X_ids_val, X_vals_val, y_click_val, y_conv_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)

print(f'Train: {len(train_ds):,}, Val: {len(val_ds):,}')
print(f'Fields: {len(all_fields)}, Dims: {field_dims}')
print(f'Train CTR: {y_click_train.mean()*100:.2f}%, CTCVR: {y_conv_train.mean()*100:.4f}%')


# ============================================================
# Models
# ============================================================
class ESMM(nn.Module):
    def __init__(self, field_dims, emb_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim, padding_idx=0) for d in field_dims])
        inp = len(field_dims) * emb_dim
        def tower(inp_dim):
            layers = []
            prev = inp_dim
            for d in hidden_dims:
                layers.extend([nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)])
                prev = d
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)
        self.ctr_tower = tower(inp)
        self.cvr_tower = tower(inp)
        self._init()

    def _init(self):
        for e in self.embeddings:
            nn.init.xavier_uniform_(e.weight.data[1:])

    def forward(self, ids, vals=None):
        embs = []
        for i, e in enumerate(self.embeddings):
            x = e(ids[:, i])
            if vals is not None:
                x = x * vals[:, i:i+1]
            embs.append(x)
        x = torch.cat(embs, dim=1)
        ctr = torch.sigmoid(self.ctr_tower(x).squeeze(-1))
        cvr = torch.sigmoid(self.cvr_tower(x).squeeze(-1))
        return ctr, cvr, ctr * cvr


class NaiveCVR(nn.Module):
    def __init__(self, field_dims, emb_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim, padding_idx=0) for d in field_dims])
        inp = len(field_dims) * emb_dim
        layers = []
        prev = inp
        for d in hidden_dims:
            layers.extend([nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)])
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids, vals=None):
        embs = []
        for i, e in enumerate(self.embeddings):
            x = e(ids[:, i])
            if vals is not None:
                x = x * vals[:, i:i+1]
            embs.append(x)
        return torch.sigmoid(self.mlp(torch.cat(embs, dim=1)).squeeze(-1))


class MMoE(nn.Module):
    def __init__(self, field_dims, emb_dim, n_experts, expert_dim, tower_dims, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim, padding_idx=0) for d in field_dims])
        inp = len(field_dims) * emb_dim
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(inp, expert_dim), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(expert_dim, expert_dim), nn.ReLU())
            for _ in range(n_experts)])
        self.gate_ctr = nn.Linear(inp, n_experts)
        self.gate_cvr = nn.Linear(inp, n_experts)
        def tower(d):
            layers = []
            prev = d
            for h in tower_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)
        self.tower_ctr = tower(expert_dim)
        self.tower_cvr = tower(expert_dim)

    def forward(self, ids, vals=None):
        embs = []
        for i, e in enumerate(self.embeddings):
            x = e(ids[:, i])
            if vals is not None:
                x = x * vals[:, i:i+1]
            embs.append(x)
        x = torch.cat(embs, dim=1)
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=1)
        g_ctr = torch.softmax(self.gate_ctr(x), dim=-1).unsqueeze(-1)
        g_cvr = torch.softmax(self.gate_cvr(x), dim=-1).unsqueeze(-1)
        ctr = torch.sigmoid(self.tower_ctr((expert_outs * g_ctr).sum(1)).squeeze(-1))
        cvr = torch.sigmoid(self.tower_cvr((expert_outs * g_cvr).sum(1)).squeeze(-1))
        return ctr, cvr, ctr * cvr


class PLELayer(nn.Module):
    def __init__(self, inp, expert_dim, n_shared, n_task, n_tasks, dropout):
        super().__init__()
        self.n_tasks = n_tasks
        total = n_shared + n_task
        self.shared = nn.ModuleList([nn.Sequential(nn.Linear(inp, expert_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_shared)])
        self.task_exp = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(inp, expert_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_task)]) for _ in range(n_tasks)])
        self.gates = nn.ModuleList([nn.Linear(inp, total) for _ in range(n_tasks)])

    def forward(self, task_inputs):
        shared_outs = [e(task_inputs[0]) for e in self.shared]
        outputs = []
        for t in range(self.n_tasks):
            task_outs = [e(task_inputs[t]) for e in self.task_exp[t]]
            all_outs = torch.stack(shared_outs + task_outs, dim=1)
            g = torch.softmax(self.gates[t](task_inputs[t]), dim=-1).unsqueeze(-1)
            outputs.append((all_outs * g).sum(1))
        return outputs


class PLE(nn.Module):
    def __init__(self, field_dims, emb_dim, n_layers, expert_dim, n_shared, n_task, tower_dims, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim, padding_idx=0) for d in field_dims])
        inp = len(field_dims) * emb_dim
        self.layers = nn.ModuleList()
        prev = inp
        for _ in range(n_layers):
            self.layers.append(PLELayer(prev, expert_dim, n_shared, n_task, 2, dropout))
            prev = expert_dim
        def tower(d):
            layers = []
            prev = d
            for h in tower_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)
        self.tower_ctr = tower(expert_dim)
        self.tower_cvr = tower(expert_dim)

    def forward(self, ids, vals=None):
        embs = []
        for i, e in enumerate(self.embeddings):
            x = e(ids[:, i])
            if vals is not None:
                x = x * vals[:, i:i+1]
            embs.append(x)
        x = torch.cat(embs, dim=1)
        task_inputs = [x, x]
        for layer in self.layers:
            task_inputs = layer(task_inputs)
        ctr = torch.sigmoid(self.tower_ctr(task_inputs[0]).squeeze(-1))
        cvr = torch.sigmoid(self.tower_cvr(task_inputs[1]).squeeze(-1))
        return ctr, cvr, ctr * cvr


# ============================================================
# Training utilities
# ============================================================
def train_epoch(model, loader, opt, dev):
    model.train()
    total = 0
    for batch in loader:
        ids, vals, click, conv = [b.to(dev) for b in batch]
        opt.zero_grad()
        ctr, cvr, ctcvr = model(ids, vals)
        loss = nn.functional.binary_cross_entropy(ctr, click) + nn.functional.binary_cross_entropy(ctcvr, conv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, dev):
    model.eval()
    all_ctr, all_cvr, all_ctcvr, all_click, all_conv = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids, vals, click, conv = [b.to(dev) for b in batch]
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
    m = {'ctr_auc': roc_auc_score(click, ctr_p)}
    m['ctcvr_auc'] = roc_auc_score(conv, ctcvr_p) if conv.sum() > 0 else 0
    clicked = click == 1
    if clicked.sum() > 10 and conv[clicked].sum() > 0:
        m['cvr_auc'] = roc_auc_score(conv[clicked], cvr_p[clicked])
    else:
        m['cvr_auc'] = 0
    return m


def run_model(model, name, epochs=NUM_EPOCHS):
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best = 0
    print(f'\n{"="*60}')
    print(f'Training {name} ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'{"="*60}')
    history = []
    for ep in range(epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, opt, device)
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        if metrics['ctcvr_auc'] > best:
            best = metrics['ctcvr_auc']
        history.append(metrics)
        print(f'  Epoch {ep+1}/{epochs} ({elapsed:.1f}s) | Loss: {loss:.4f} | '
              f'CTR: {metrics["ctr_auc"]:.4f} | CVR: {metrics["cvr_auc"]:.4f} | '
              f'CTCVR: {metrics["ctcvr_auc"]:.4f}')
    return {'ctr_auc': max(h['ctr_auc'] for h in history),
            'cvr_auc': max(h['cvr_auc'] for h in history),
            'ctcvr_auc': best,
            'params': sum(p.numel() for p in model.parameters())}


# ============================================================
# Run all models
# ============================================================
results = {}

# 1. Naive CVR
print(f'\n{"="*60}')
print('Training Naive CVR (clicked samples only)')
print(f'{"="*60}')
naive = NaiveCVR(field_dims, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
naive_opt = optim.Adam(naive.parameters(), lr=LR, weight_decay=1e-5)
click_mask = data['y_click_train'] == 1
X_c = torch.LongTensor(data['X_ids_train'][click_mask])
V_c = torch.FloatTensor(data['X_vals_train'][click_mask])
y_c = torch.FloatTensor(data['y_conv_train'][click_mask])
clicked_loader = DataLoader(TensorDataset(X_c, V_c, y_c), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f'Clicked samples: {len(X_c):,}')

for ep in range(NUM_EPOCHS):
    naive.train()
    total = 0
    for ids, vals, conv in clicked_loader:
        ids, vals, conv = ids.to(device), vals.to(device), conv.to(device)
        naive_opt.zero_grad()
        pred = naive(ids, vals)
        loss = nn.functional.binary_cross_entropy(pred, conv)
        loss.backward()
        naive_opt.step()
        total += loss.item()
    print(f'  Epoch {ep+1}: Loss = {total/len(clicked_loader):.4f}')

naive.eval()
preds = []
with torch.no_grad():
    for batch in val_loader:
        ids, vals = batch[0].to(device), batch[1].to(device)
        preds.append(naive(ids, vals).cpu().numpy())
naive_preds = np.concatenate(preds)
click_val = data['y_click_val']
conv_val = data['y_conv_val']
naive_ctcvr = roc_auc_score(conv_val, naive_preds)
clicked_val = click_val == 1
naive_cvr = roc_auc_score(conv_val[clicked_val], naive_preds[clicked_val]) if clicked_val.sum() > 0 and conv_val[clicked_val].sum() > 0 else 0
print(f'Naive CVR: CVR AUC={naive_cvr:.4f}, CTCVR AUC={naive_ctcvr:.4f}')
results['naive_cvr'] = {'cvr_auc': float(naive_cvr), 'ctcvr_auc': float(naive_ctcvr)}

# 2. ESMM
esmm = ESMM(field_dims, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
results['esmm'] = run_model(esmm, 'ESMM')

# 3. MMoE
mmoe = MMoE(field_dims, EMBEDDING_DIM, N_EXPERTS, 128, [64, 32], DROPOUT).to(device)
results['mmoe'] = run_model(mmoe, 'MMoE')

# 4. PLE
ple = PLE(field_dims, EMBEDDING_DIM, 2, 128, 2, 2, [64, 32], DROPOUT).to(device)
results['ple'] = run_model(ple, 'PLE')

# ============================================================
# Summary
# ============================================================
print(f'\n{"="*60}')
print('FINAL RESULTS: Ali-CCP Multi-Task CVR')
print(f'{"="*60}')
print(f'{"Model":<15} {"CTR AUC":>10} {"CVR AUC":>10} {"CTCVR AUC":>10}')
print('-' * 50)
for name, res in results.items():
    ctr = f'{res.get("ctr_auc", 0):.4f}' if res.get('ctr_auc') else '-'
    cvr = f'{res["cvr_auc"]:.4f}'
    ctcvr = f'{res["ctcvr_auc"]:.4f}'
    print(f'{name:<15} {ctr:>10} {cvr:>10} {ctcvr:>10}')

# Save results
with open(PROCESSED_DIR / 'aliccp_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Also save for comparison notebook
with open(PROCESSED_DIR / 'esmm_results.json', 'w') as f:
    json.dump({'esmm': results['esmm'], 'naive_cvr': results['naive_cvr']}, f, indent=2)
with open(PROCESSED_DIR / 'advanced_cvr_results.json', 'w') as f:
    json.dump({'mmoe': results['mmoe'], 'ple': results['ple']}, f, indent=2)

print(f'\nResults saved to {PROCESSED_DIR / "aliccp_results.json"}')
print('Done!')
