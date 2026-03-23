"""Run BST ablation study only."""
import numpy as np
import pickle
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'taobao', 'processed')

print('\nLoading data...')
with open(os.path.join(PROCESSED_DIR, 'taobao_sequential_data.pkl'), 'rb') as f:
    data = pickle.load(f)

user_sequences = data['user_sequences']
n_items = data['n_items']
n_categories = data['n_categories']
item_popularity = data['item_popularity']

item_to_cat = {}
for uid, seq in user_sequences.items():
    for item_id, cat_id in zip(seq['item_ids'], seq['cat_ids']):
        item_to_cat[item_id] = cat_id

all_item_ids = list(item_popularity.keys())
all_item_probs = np.array([item_popularity[i] for i in all_item_ids])
all_item_probs = all_item_probs / all_item_probs.sum()

MAX_SEQ_LEN = 50
NEG_RATIO = 4
BATCH_SIZE = 1024
MAX_EPOCHS = 15
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6

print(f'Loaded: {len(user_sequences)} users, {n_items} items')


class TaobaoCTRDataset(Dataset):
    def __init__(self, user_sequences, item_to_cat, all_item_ids, all_item_probs,
                 max_seq_len=50, neg_ratio=4, mode='train'):
        self.max_seq_len = max_seq_len
        self.neg_ratio = neg_ratio
        self.item_to_cat = item_to_cat
        self.all_item_ids = np.array(all_item_ids)
        self.all_item_probs = all_item_probs
        self.samples = []
        n_users = len(user_sequences)
        pool_size = n_users * neg_ratio * 3
        print(f'  Pre-sampling {pool_size:,} negatives...')
        neg_pool = np.random.choice(self.all_item_ids, size=pool_size, p=self.all_item_probs)
        pool_idx = 0
        for i, (uid, seq) in enumerate(user_sequences.items()):
            if i % 20000 == 0: print(f'  User {i}/{n_users}...')
            items = seq['item_ids']
            cats = seq['cat_ids']
            if len(items) < 3: continue
            target_idx = len(items) - 2 if mode == 'train' else len(items) - 1
            hist_items = items[:target_idx]
            hist_cats = cats[:target_idx]
            if not hist_items: continue
            if len(hist_items) > max_seq_len:
                hist_items = hist_items[-max_seq_len:]
                hist_cats = hist_cats[-max_seq_len:]
            hist_len = len(hist_items)
            pad_len = max_seq_len - hist_len
            hi = [0]*pad_len + hist_items
            hc = [0]*pad_len + hist_cats
            self.samples.append((hi, hc, hist_len, items[target_idx], cats[target_idx], 1))
            user_item_set = set(items)
            nc = 0
            while nc < neg_ratio:
                if pool_idx >= len(neg_pool):
                    neg_pool = np.random.choice(self.all_item_ids, size=pool_size, p=self.all_item_probs)
                    pool_idx = 0
                ni = neg_pool[pool_idx]; pool_idx += 1
                if ni not in user_item_set:
                    self.samples.append((hi, hc, hist_len, ni, item_to_cat.get(ni, 0), 0))
                    nc += 1
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        hi, hc, hl, ti, tc, l = self.samples[idx]
        return {'hist_items': torch.LongTensor(hi), 'hist_cats': torch.LongTensor(hc),
                'hist_len': torch.LongTensor([hl]), 'target_item': torch.LongTensor([ti]),
                'target_cat': torch.LongTensor([tc]), 'label': torch.FloatTensor([l])}


class BST(nn.Module):
    def __init__(self, n_items, n_categories, embed_dim=32, n_heads=4, n_layers=2,
                 max_seq_len=50, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        cat_dim = embed_dim // 2
        self.feature_dim = embed_dim + cat_dim
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(n_categories, cat_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, self.feature_dim)
        self.emb_layernorm = nn.LayerNorm(self.feature_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=n_heads, dim_feedforward=self.feature_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        layers = []
        in_dim = self.feature_dim * 2
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])
    def forward(self, hist_items, hist_cats, hist_len, target_item, target_cat, **kw):
        B, S = hist_items.shape
        ie = self.item_embedding(hist_items)
        ce = self.cat_embedding(hist_cats)
        se = torch.cat([ie, ce], dim=-1)
        pos = torch.arange(S, device=hist_items.device).unsqueeze(0).expand(B, S)
        se = se + self.position_embedding(pos)
        se = self.emb_dropout(self.emb_layernorm(se))
        pm = (hist_items == 0)
        to = self.transformer_encoder(se, src_key_padding_mask=pm)
        mf = (~pm).float().unsqueeze(-1)
        pooled = (to * mf).sum(1) / mf.sum(1).clamp(min=1)
        te = torch.cat([self.item_embedding(target_item).squeeze(1),
                        self.cat_embedding(target_cat).squeeze(1)], dim=-1)
        return self.mlp(torch.cat([pooled, te], dim=-1)), None


def train_epoch(model, loader, optimizer, criterion):
    model.train(); total_loss = 0; n = 0
    for b in loader:
        optimizer.zero_grad()
        logits, _ = model(b['hist_items'].to(device), b['hist_cats'].to(device),
                          b['hist_len'].to(device), b['target_item'].to(device), b['target_cat'].to(device))
        loss = criterion(logits, b['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / n

def evaluate(model, loader, criterion):
    model.eval(); total_loss = 0; n = 0; labels = []; preds = []
    with torch.no_grad():
        for b in loader:
            logits, _ = model(b['hist_items'].to(device), b['hist_cats'].to(device),
                              b['hist_len'].to(device), b['target_item'].to(device), b['target_cat'].to(device))
            loss = criterion(logits, b['label'].to(device))
            total_loss += loss.item(); n += 1
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(b['label'].numpy().flatten())
    return total_loss/n, roc_auc_score(labels, preds), log_loss(labels, np.clip(preds, 1e-7, 1-1e-7))


print('\nBuilding CTR datasets...')
train_ds = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs, MAX_SEQ_LEN, NEG_RATIO, 'train')
test_ds = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs, MAX_SEQ_LEN, NEG_RATIO, 'test')
print(f'Train: {len(train_ds):,}, Test: {len(test_ds):,}')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
configs = [
    {'n_heads': 1, 'n_layers': 1, 'name': '1H-1L'},
    {'n_heads': 4, 'n_layers': 1, 'name': '4H-1L'},
    {'n_heads': 4, 'n_layers': 2, 'name': '4H-2L'},
]

results = {}
for cfg in configs:
    print(f"\nTraining BST ({cfg['name']})...")
    m = BST(n_items, n_categories, 32, cfg['n_heads'], cfg['n_layers'], MAX_SEQ_LEN, [256,128,64], 0.2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = 0; pc = 0
    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tl = train_epoch(m, train_loader, opt, criterion)
        _, auc, _ = evaluate(m, test_loader, criterion)
        el = time.time() - t0
        imp = ' *' if auc > best else ''
        print(f"  Epoch {ep+1:2d} | Loss: {tl:.4f} | AUC: {auc:.4f} | Time: {el:.1f}s{imp}")
        if auc > best: best = auc; pc = 0
        else: pc += 1
        if pc >= PATIENCE: break
    results[cfg['name']] = best
    print(f"  {cfg['name']}: Best AUC = {best:.4f}")

print('\n' + '='*50)
print('BST Ablation Results:')
for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f'  {name:10s}: AUC = {auc:.4f}')

# Save
rp = os.path.join(PROCESSED_DIR, 'transformer_results.json')
r = {}
if os.path.exists(rp):
    with open(rp) as f: r = json.load(f)
r['ablation_results'] = results
with open(rp, 'w') as f: json.dump(r, f, indent=2)
print(f'\nSaved to {rp}')
