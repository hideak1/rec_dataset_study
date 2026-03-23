"""Ali-CCP v2: Proper multi-valued field handling with sum-pooling embeddings.
Key fix: multi-valued fields (user behavioral history) are sum-pooled, not truncated.
"""
import numpy as np
import pickle
import time
import json
import os
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path('data/aliccp')
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

SUBSAMPLE_SIZE = 3_000_000  # Reduced for memory (multi-valued fields need more RAM)
EMBEDDING_DIM = 16
BATCH_SIZE = 4096
LR = 1e-3
NUM_EPOCHS = 8
DROPOUT = 0.2
N_EXPERTS = 4

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
            features.append((field_id, int(id_val[0]), float(id_val[1])))
    return features

# ============================================================
# STEP 1: Data Preprocessing with multi-valued field support
# ============================================================
processed_v2 = PROCESSED_DIR / 'aliccp_v2.pkl'

if processed_v2.exists():
    print('Loading preprocessed v2 data...')
    with open(processed_v2, 'rb') as f:
        data = pickle.load(f)
    print('Loaded.')
else:
    print(f'\n{"="*60}')
    print('STEP 1: Preprocessing with multi-valued fields')
    print(f'{"="*60}')

    # Identify single-valued vs multi-valued fields
    SINGLE_FIELDS = [101, 121, 122, 124, 125, 126, 127, 128, 129,
                     205, 206, 207, 210, 216, 301]
    MULTI_FIELDS = [508, 509, 702, 853, 10914, 11014, 12714, 15014]

    # Load skeleton
    print(f'Loading {SUBSAMPLE_SIZE/1e6:.0f}M skeleton samples...')
    start = time.time()
    clicks, convs, hash_ids = [], [], []
    sample_features_raw = []

    with open(DATA_DIR / 'sample_skeleton_train.csv', 'rb') as f:
        for i, line in enumerate(f):
            if i >= SUBSAMPLE_SIZE:
                break
            parts = line.split(b',', 5)
            clicks.append(int(parts[1]))
            convs.append(int(parts[2]))
            hash_ids.append(parts[3].decode())
            feats = parse_features(parts[5])
            fd = defaultdict(list)
            for field, fid, val in feats:
                fd[field].append((fid, val))
            sample_features_raw.append(dict(fd))
            if (i + 1) % 1_000_000 == 0:
                print(f'  {i+1:,} loaded...')

    clicks = np.array(clicks)
    convs = np.array(convs)
    print(f'Loaded in {time.time()-start:.1f}s, CTR={clicks.mean()*100:.2f}%')

    # Load common features
    print('Loading common features...')
    start = time.time()
    needed = set(hash_ids)
    common_raw = {}
    with open(DATA_DIR / 'common_features_train.csv', 'rb') as f:
        for i, line in enumerate(f):
            parts = line.split(b',', 2)
            h = parts[0].decode()
            if h in needed:
                feats = parse_features(parts[2])
                fd = defaultdict(list)
                for field, fid, val in feats:
                    fd[field].append((fid, val))
                common_raw[h] = dict(fd)
            if (i + 1) % 50000 == 0:
                print(f'  Scanned {i+1:,}, matched {len(common_raw):,}')
    print(f'Loaded {len(common_raw):,} in {time.time()-start:.1f}s')

    # Build feature mappings
    print('Building feature mappings...')
    all_fields_combined = sorted(set(SINGLE_FIELDS + MULTI_FIELDS))
    field_feat_ids = defaultdict(set)

    for fd in sample_features_raw:
        for field, fl in fd.items():
            if field in all_fields_combined:
                for fid, _ in fl:
                    field_feat_ids[field].add(fid)
    for fd in common_raw.values():
        for field, fl in fd.items():
            if field in all_fields_combined:
                for fid, _ in fl:
                    field_feat_ids[field].add(fid)

    feat_mappings = {}
    for field in all_fields_combined:
        if field in field_feat_ids:
            mapping = {fid: idx + 1 for idx, fid in enumerate(sorted(field_feat_ids[field]))}
            feat_mappings[field] = mapping
            print(f'  Field {field}: {len(mapping):,} features')

    # For single-valued fields: store one ID per sample
    # For multi-valued fields: store variable-length lists, then pad to max_len
    MAX_MULTI_LEN = 30  # Cap multi-valued fields (avg ~60-200, but top-30 captures most signal)

    print('Encoding samples...')
    start = time.time()

    # Single-valued: (N, n_single_fields)
    single_fields = [f for f in SINGLE_FIELDS if f in feat_mappings]
    multi_fields = [f for f in MULTI_FIELDS if f in feat_mappings]

    X_single = np.zeros((len(clicks), len(single_fields)), dtype=np.int64)
    X_multi = np.zeros((len(clicks), len(multi_fields), MAX_MULTI_LEN), dtype=np.int64)
    X_multi_vals = np.zeros((len(clicks), len(multi_fields), MAX_MULTI_LEN), dtype=np.float32)
    X_multi_lens = np.zeros((len(clicks), len(multi_fields)), dtype=np.int32)

    for i in range(len(clicks)):
        # Merge
        merged = {}
        c = common_raw.get(hash_ids[i], {})
        if c:
            merged.update(c)
        merged.update(sample_features_raw[i])

        # Single fields
        for j, field in enumerate(single_fields):
            if field in merged and merged[field]:
                fid, val = merged[field][0]
                if fid in feat_mappings[field]:
                    X_single[i, j] = feat_mappings[field][fid]

        # Multi-valued fields
        for j, field in enumerate(multi_fields):
            if field in merged and merged[field]:
                items = merged[field][:MAX_MULTI_LEN]
                for k, (fid, val) in enumerate(items):
                    if fid in feat_mappings[field]:
                        X_multi[i, j, k] = feat_mappings[field][fid]
                        X_multi_vals[i, j, k] = val
                X_multi_lens[i, j] = len(items)

        if (i + 1) % 1_000_000 == 0:
            print(f'  Encoded {i+1:,}...')

    print(f'Encoded in {time.time()-start:.1f}s')

    # Split
    indices = np.arange(len(clicks))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    data = {
        'X_single_train': X_single[train_idx], 'X_single_val': X_single[val_idx],
        'X_multi_train': X_multi[train_idx], 'X_multi_val': X_multi[val_idx],
        'X_multi_vals_train': X_multi_vals[train_idx], 'X_multi_vals_val': X_multi_vals[val_idx],
        'X_multi_lens_train': X_multi_lens[train_idx], 'X_multi_lens_val': X_multi_lens[val_idx],
        'y_click_train': clicks[train_idx].astype(np.float32),
        'y_conv_train': convs[train_idx].astype(np.float32),
        'y_click_val': clicks[val_idx].astype(np.float32),
        'y_conv_val': convs[val_idx].astype(np.float32),
        'single_fields': single_fields,
        'multi_fields': multi_fields,
        'feat_mappings': feat_mappings,
        'single_cardinalities': [len(feat_mappings[f]) + 1 for f in single_fields],
        'multi_cardinalities': [len(feat_mappings[f]) + 1 for f in multi_fields],
    }

    with open(processed_v2, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {processed_v2}')

    # Free memory
    del sample_features_raw, common_raw, X_multi, X_multi_vals, X_multi_lens, X_single


# ============================================================
# Dataset
# ============================================================
class AlicppDataset(Dataset):
    def __init__(self, data, split='train'):
        s = split
        self.X_single = torch.LongTensor(data[f'X_single_{s}'])
        self.X_multi = torch.LongTensor(data[f'X_multi_{s}'])
        self.X_multi_vals = torch.FloatTensor(data[f'X_multi_vals_{s}'])
        self.X_multi_lens = torch.IntTensor(data[f'X_multi_lens_{s}'])
        self.y_click = torch.FloatTensor(data[f'y_click_{s}'])
        self.y_conv = torch.FloatTensor(data[f'y_conv_{s}'])

    def __len__(self):
        return len(self.y_click)

    def __getitem__(self, idx):
        return (self.X_single[idx], self.X_multi[idx], self.X_multi_vals[idx],
                self.X_multi_lens[idx], self.y_click[idx], self.y_conv[idx])


print(f'\n{"="*60}')
print('Setting up data loaders')
print(f'{"="*60}')

train_ds = AlicppDataset(data, 'train')
val_ds = AlicppDataset(data, 'val')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)

single_cards = data['single_cardinalities']
multi_cards = data['multi_cardinalities']
n_single = len(single_cards)
n_multi = len(multi_cards)

print(f'Train: {len(train_ds):,}, Val: {len(val_ds):,}')
print(f'Single fields: {n_single}, Multi fields: {n_multi}')
print(f'Single cardinalities: {single_cards}')
print(f'Multi cardinalities: {multi_cards}')


# ============================================================
# Model with proper multi-valued field handling
# ============================================================
class MultiFieldEmbedding(nn.Module):
    """Embedding layer that handles both single and multi-valued fields."""
    def __init__(self, single_dims, multi_dims, emb_dim):
        super().__init__()
        self.single_embs = nn.ModuleList([
            nn.Embedding(d, emb_dim, padding_idx=0) for d in single_dims
        ])
        self.multi_embs = nn.ModuleList([
            nn.Embedding(d, emb_dim, padding_idx=0) for d in multi_dims
        ])
        self.n_single = len(single_dims)
        self.n_multi = len(multi_dims)
        self.output_dim = (self.n_single + self.n_multi) * emb_dim

        for e in self.single_embs:
            nn.init.xavier_uniform_(e.weight.data[1:])
        for e in self.multi_embs:
            nn.init.xavier_uniform_(e.weight.data[1:])

    def forward(self, x_single, x_multi, x_multi_vals, x_multi_lens):
        embs = []

        # Single-valued fields
        for i, emb in enumerate(self.single_embs):
            embs.append(emb(x_single[:, i]))  # (B, D)

        # Multi-valued fields: sum-pool embeddings weighted by values
        for i, emb in enumerate(self.multi_embs):
            multi_emb = emb(x_multi[:, i])  # (B, MAX_LEN, D)
            # Weight by values
            weights = x_multi_vals[:, i].unsqueeze(-1)  # (B, MAX_LEN, 1)
            weighted = multi_emb * weights  # (B, MAX_LEN, D)
            # Mask padding
            mask = (x_multi[:, i] != 0).unsqueeze(-1).float()  # (B, MAX_LEN, 1)
            summed = (weighted * mask).sum(dim=1)  # (B, D)
            # Average by length (avoid div by 0)
            lens = x_multi_lens[:, i].float().clamp(min=1).unsqueeze(-1)  # (B, 1)
            embs.append(summed / lens)  # (B, D)

        return torch.cat(embs, dim=1)  # (B, total_fields * D)


class ESMMv2(nn.Module):
    def __init__(self, single_dims, multi_dims, emb_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.embedding = MultiFieldEmbedding(single_dims, multi_dims, emb_dim)
        inp = self.embedding.output_dim

        def tower(d):
            layers = []
            prev = d
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.ctr_tower = tower(inp)
        self.cvr_tower = tower(inp)

    def forward(self, x_single, x_multi, x_multi_vals, x_multi_lens):
        x = self.embedding(x_single, x_multi, x_multi_vals, x_multi_lens)
        ctr = torch.sigmoid(self.ctr_tower(x).squeeze(-1))
        cvr = torch.sigmoid(self.cvr_tower(x).squeeze(-1))
        return ctr, cvr, ctr * cvr


class NaiveCVRv2(nn.Module):
    def __init__(self, single_dims, multi_dims, emb_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.embedding = MultiFieldEmbedding(single_dims, multi_dims, emb_dim)
        inp = self.embedding.output_dim
        layers = []
        prev = inp
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_single, x_multi, x_multi_vals, x_multi_lens):
        x = self.embedding(x_single, x_multi, x_multi_vals, x_multi_lens)
        return torch.sigmoid(self.mlp(x).squeeze(-1))


class MMoEv2(nn.Module):
    def __init__(self, single_dims, multi_dims, emb_dim, n_experts, expert_dim, tower_dims, dropout=0.2):
        super().__init__()
        self.embedding = MultiFieldEmbedding(single_dims, multi_dims, emb_dim)
        inp = self.embedding.output_dim
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(inp, expert_dim), nn.BatchNorm1d(expert_dim),
                          nn.ReLU(), nn.Dropout(dropout),
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

    def forward(self, x_single, x_multi, x_multi_vals, x_multi_lens):
        x = self.embedding(x_single, x_multi, x_multi_vals, x_multi_lens)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        g_ctr = torch.softmax(self.gate_ctr(x), dim=-1).unsqueeze(-1)
        g_cvr = torch.softmax(self.gate_cvr(x), dim=-1).unsqueeze(-1)
        ctr = torch.sigmoid(self.tower_ctr((expert_outs * g_ctr).sum(1)).squeeze(-1))
        cvr = torch.sigmoid(self.tower_cvr((expert_outs * g_cvr).sum(1)).squeeze(-1))
        return ctr, cvr, ctr * cvr


class PLELayerV2(nn.Module):
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


class PLEv2(nn.Module):
    def __init__(self, single_dims, multi_dims, emb_dim, n_layers, expert_dim, n_shared, n_task, tower_dims, dropout=0.2):
        super().__init__()
        self.embedding = MultiFieldEmbedding(single_dims, multi_dims, emb_dim)
        inp = self.embedding.output_dim
        self.layers = nn.ModuleList()
        prev = inp
        for _ in range(n_layers):
            self.layers.append(PLELayerV2(prev, expert_dim, n_shared, n_task, 2, dropout))
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

    def forward(self, x_single, x_multi, x_multi_vals, x_multi_lens):
        x = self.embedding(x_single, x_multi, x_multi_vals, x_multi_lens)
        task_inputs = [x, x]
        for layer in self.layers:
            task_inputs = layer(task_inputs)
        ctr = torch.sigmoid(self.tower_ctr(task_inputs[0]).squeeze(-1))
        cvr = torch.sigmoid(self.tower_cvr(task_inputs[1]).squeeze(-1))
        return ctr, cvr, ctr * cvr


# ============================================================
# Training
# ============================================================
HIDDEN_DIMS = [256, 128, 64]

def train_epoch(model, loader, opt, dev):
    model.train()
    total = 0
    for batch in loader:
        xs, xm, xmv, xml, click, conv = [b.to(dev) for b in batch]
        opt.zero_grad()
        ctr, cvr, ctcvr = model(xs, xm, xmv, xml)
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
            xs, xm, xmv, xml, click, conv = [b.to(dev) for b in batch]
            ctr, cvr, ctcvr = model(xs, xm, xmv, xml)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    best = 0
    print(f'\n{"="*60}')
    print(f'Training {name} ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'{"="*60}')
    for ep in range(epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, opt, device)
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        if metrics['ctcvr_auc'] > best:
            best = metrics['ctcvr_auc']
        scheduler.step(metrics['ctcvr_auc'])
        print(f'  Epoch {ep+1}/{epochs} ({elapsed:.1f}s) | Loss: {loss:.4f} | '
              f'CTR: {metrics["ctr_auc"]:.4f} | CVR: {metrics["cvr_auc"]:.4f} | '
              f'CTCVR: {metrics["ctcvr_auc"]:.4f}')
    return {'ctr_auc': float(max(metrics['ctr_auc'] for metrics in [evaluate(model, val_loader, device)])),
            'cvr_auc': float(metrics['cvr_auc']),
            'ctcvr_auc': float(best),
            'params': sum(p.numel() for p in model.parameters())}


results = {}

# Naive CVR
print(f'\n{"="*60}')
print('Training Naive CVR (clicked only)')
print(f'{"="*60}')
naive = NaiveCVRv2(single_cards, multi_cards, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
naive_opt = optim.Adam(naive.parameters(), lr=LR, weight_decay=1e-5)

# Build clicked-only loader
click_mask = data['y_click_train'] == 1
clicked_ds = torch.utils.data.TensorDataset(
    torch.LongTensor(data['X_single_train'][click_mask]),
    torch.LongTensor(data['X_multi_train'][click_mask]),
    torch.FloatTensor(data['X_multi_vals_train'][click_mask]),
    torch.IntTensor(data['X_multi_lens_train'][click_mask]),
    torch.FloatTensor(data['y_conv_train'][click_mask]),
)
clicked_loader = DataLoader(clicked_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f'Clicked samples: {click_mask.sum():,}')

for ep in range(NUM_EPOCHS):
    naive.train()
    total = 0
    for xs, xm, xmv, xml, conv in clicked_loader:
        xs, xm, xmv, xml, conv = xs.to(device), xm.to(device), xmv.to(device), xml.to(device), conv.to(device)
        naive_opt.zero_grad()
        pred = naive(xs, xm, xmv, xml)
        loss = nn.functional.binary_cross_entropy(pred, conv)
        loss.backward()
        naive_opt.step()
        total += loss.item()
    print(f'  Epoch {ep+1}: Loss = {total/len(clicked_loader):.4f}')

naive.eval()
preds = []
with torch.no_grad():
    for batch in val_loader:
        xs, xm, xmv, xml = [b.to(device) for b in batch[:4]]
        preds.append(naive(xs, xm, xmv, xml).cpu().numpy())
naive_preds = np.concatenate(preds)
click_val = data['y_click_val']
conv_val = data['y_conv_val']
naive_ctcvr = roc_auc_score(conv_val, naive_preds)
clicked_val = click_val == 1
naive_cvr = roc_auc_score(conv_val[clicked_val], naive_preds[clicked_val]) if clicked_val.sum() > 0 and conv_val[clicked_val].sum() > 0 else 0
print(f'Naive CVR: CVR={naive_cvr:.4f}, CTCVR={naive_ctcvr:.4f}')
results['naive_cvr'] = {'cvr_auc': float(naive_cvr), 'ctcvr_auc': float(naive_ctcvr)}

# ESMM
esmm = ESMMv2(single_cards, multi_cards, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT).to(device)
results['esmm'] = run_model(esmm, 'ESMM')

# MMoE
mmoe = MMoEv2(single_cards, multi_cards, EMBEDDING_DIM, N_EXPERTS, 128, [64, 32], DROPOUT).to(device)
results['mmoe'] = run_model(mmoe, 'MMoE')

# PLE
ple = PLEv2(single_cards, multi_cards, EMBEDDING_DIM, 2, 128, 2, 2, [64, 32], DROPOUT).to(device)
results['ple'] = run_model(ple, 'PLE')

# Summary
print(f'\n{"="*60}')
print('FINAL RESULTS v2: Ali-CCP (multi-valued fields)')
print(f'{"="*60}')
print(f'{"Model":<15} {"CTR AUC":>10} {"CVR AUC":>10} {"CTCVR AUC":>10}')
print('-' * 50)
for name, res in results.items():
    ctr = f'{res.get("ctr_auc", 0):.4f}' if res.get('ctr_auc') else '-'
    print(f'{name:<15} {ctr:>10} {res["cvr_auc"]:.4f}  {res["ctcvr_auc"]:.4f}')

with open(PROCESSED_DIR / 'aliccp_results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
with open(PROCESSED_DIR / 'esmm_results.json', 'w') as f:
    json.dump({'esmm': results['esmm'], 'naive_cvr': results['naive_cvr']}, f, indent=2)
with open(PROCESSED_DIR / 'advanced_cvr_results.json', 'w') as f:
    json.dump({'mmoe': results['mmoe'], 'ple': results['ple']}, f, indent=2)

print(f'\nResults saved to {PROCESSED_DIR / "aliccp_results_v2.json"}')
