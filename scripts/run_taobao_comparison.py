"""Run Taobao comparison: Mean Pooling, DIN, DIEN, BST on same data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
NB_DIR = os.path.join(BASE_DIR, 'notebooks', '02_taobao_sequential')

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

# ===== Dataset with batch pre-sampling =====
class TaobaoCTRDataset(Dataset):
    def __init__(self, user_sequences, item_to_cat, all_item_ids, all_item_probs,
                 max_seq_len=50, neg_ratio=4, mode='train'):
        self.samples = []
        all_item_ids = np.array(all_item_ids)
        n_users = len(user_sequences)
        pool_size = n_users * neg_ratio * 3
        print(f'  Pre-sampling {pool_size:,} negatives...')
        neg_pool = np.random.choice(all_item_ids, size=pool_size, p=all_item_probs)
        pool_idx = 0
        for i, (uid, seq) in enumerate(user_sequences.items()):
            if i % 20000 == 0: print(f'  User {i}/{n_users}...')
            items = seq['item_ids']; cats = seq['cat_ids']
            if len(items) < 3: continue
            ti = len(items) - 2 if mode == 'train' else len(items) - 1
            hi, hc = items[:ti], cats[:ti]
            if not hi: continue
            if len(hi) > max_seq_len: hi, hc = hi[-max_seq_len:], hc[-max_seq_len:]
            hl = len(hi); pl = max_seq_len - hl
            hip, hcp = [0]*pl + hi, [0]*pl + hc
            self.samples.append((hip, hcp, hl, items[ti], cats[ti], 1))
            uis = set(items); nc = 0
            while nc < neg_ratio:
                if pool_idx >= len(neg_pool):
                    neg_pool = np.random.choice(all_item_ids, size=pool_size, p=all_item_probs)
                    pool_idx = 0
                ni = neg_pool[pool_idx]; pool_idx += 1
                if ni not in uis:
                    self.samples.append((hip, hcp, hl, ni, item_to_cat.get(ni, 0), 0)); nc += 1
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        hi, hc, hl, ti, tc, l = self.samples[idx]
        return {'hist_items': torch.LongTensor(hi), 'hist_cats': torch.LongTensor(hc),
                'hist_len': torch.LongTensor([hl]), 'target_item': torch.LongTensor([ti]),
                'target_cat': torch.LongTensor([tc]), 'label': torch.FloatTensor([l])}

print('Building datasets...')
train_ds = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs, MAX_SEQ_LEN, NEG_RATIO, 'train')
test_ds = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs, MAX_SEQ_LEN, NEG_RATIO, 'test')
print(f'Train: {len(train_ds):,}, Test: {len(test_ds):,}')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ===== Models =====
class Dice(nn.Module):
    def __init__(self, n, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(n, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        p = torch.sigmoid(self.bn(x))
        return p * x + (1-p) * self.alpha * x

class MeanPoolingModel(nn.Module):
    def __init__(self, n_items, n_cats, ed=32, hd=[256,128,64], dr=0.2):
        super().__init__()
        self.ie = nn.Embedding(n_items, ed, padding_idx=0)
        self.ce = nn.Embedding(n_cats, ed//2, padding_idx=0)
        fd = ed + ed//2; layers = []; ind = fd*2
        for h in hd:
            layers.extend([nn.Linear(ind,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr)]); ind=h
        layers.append(nn.Linear(ind,1)); self.mlp = nn.Sequential(*layers)
    def forward(self, hi, hc, hl, ti, tc):
        he = torch.cat([self.ie(hi), self.ce(hc)], -1)
        te = torch.cat([self.ie(ti).squeeze(1), self.ce(tc).squeeze(1)], -1)
        m = (hi != 0).float().unsqueeze(-1)
        u = (he * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(torch.cat([u, te], -1)), None

class DINAttention(nn.Module):
    def __init__(self, ed, ah=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ed*4, ah), Dice(ah), nn.Linear(ah, ah//2), Dice(ah//2), nn.Linear(ah//2, 1))
    def forward(self, he, te, m):
        tex = te.expand_as(he)
        ai = torch.cat([he, tex, he-tex, he*tex], -1)
        B, S, D = ai.shape
        a = self.mlp(ai.view(B*S, D)).view(B, S) * m.float()
        return (he * a.unsqueeze(-1)).sum(1), a

class DIN(nn.Module):
    def __init__(self, n_items, n_cats, ed=32, ah=64, hd=[256,128,64], dr=0.2):
        super().__init__()
        self.ie = nn.Embedding(n_items, ed, padding_idx=0)
        self.ce = nn.Embedding(n_cats, ed//2, padding_idx=0)
        fd = ed + ed//2; self.att = DINAttention(fd, ah)
        layers = []; ind = fd*2
        for h in hd:
            layers.extend([nn.Linear(ind,h), nn.BatchNorm1d(h), Dice(h), nn.Dropout(dr)]); ind=h
        layers.append(nn.Linear(ind,1)); self.mlp = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])
    def forward(self, hi, hc, hl, ti, tc):
        he = torch.cat([self.ie(hi), self.ce(hc)], -1)
        te = torch.cat([self.ie(ti), self.ce(tc)], -1)
        m = (hi != 0)
        u, aw = self.att(he, te, m)
        return self.mlp(torch.cat([u, te.squeeze(1)], -1)), aw

class DIEN(nn.Module):
    def __init__(self, n_items, n_cats, ed=32, gh=64, hd=[256,128,64], dr=0.2):
        super().__init__()
        self.ie = nn.Embedding(n_items, ed, padding_idx=0)
        self.ce = nn.Embedding(n_cats, ed//2, padding_idx=0)
        fd = ed + ed//2
        self.gru = nn.GRU(fd, gh, batch_first=True)
        self.att = DINAttention(gh, gh)
        self.tp = nn.Linear(fd, gh)
        layers = []; ind = gh*2
        for h in hd:
            layers.extend([nn.Linear(ind,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr)]); ind=h
        layers.append(nn.Linear(ind,1)); self.mlp = nn.Sequential(*layers)
    def forward(self, hi, hc, hl, ti, tc):
        he = torch.cat([self.ie(hi), self.ce(hc)], -1)
        go, _ = self.gru(he)
        te = torch.cat([self.ie(ti), self.ce(tc)], -1)
        tp = self.tp(te.squeeze(1)).unsqueeze(1)
        m = (hi != 0)
        u, aw = self.att(go, tp, m)
        return self.mlp(torch.cat([u, tp.squeeze(1)], -1)), aw

class BST(nn.Module):
    def __init__(self, n_items, n_cats, ed=32, nh=4, nl=2, ms=50, hd=[256,128,64], dr=0.2):
        super().__init__()
        cd = ed//2; self.fd = ed+cd
        self.ie = nn.Embedding(n_items, ed, padding_idx=0)
        self.ce = nn.Embedding(n_cats, cd, padding_idx=0)
        self.pe = nn.Embedding(ms, self.fd)
        self.eln = nn.LayerNorm(self.fd); self.edr = nn.Dropout(dr)
        el = nn.TransformerEncoderLayer(d_model=self.fd, nhead=nh, dim_feedforward=self.fd*4,
                                         dropout=dr, activation='gelu', batch_first=True, norm_first=True)
        self.te = nn.TransformerEncoder(el, num_layers=nl)
        layers = []; ind = self.fd*2
        for h in hd:
            layers.extend([nn.Linear(ind,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dr)]); ind=h
        layers.append(nn.Linear(ind,1)); self.mlp = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])
    def forward(self, hi, hc, hl, ti, tc):
        B, S = hi.shape
        se = torch.cat([self.ie(hi), self.ce(hc)], -1)
        p = torch.arange(S, device=hi.device).unsqueeze(0).expand(B, S)
        se = self.edr(self.eln(se + self.pe(p)))
        pm = (hi == 0)
        to = self.te(se, src_key_padding_mask=pm)
        mf = (~pm).float().unsqueeze(-1)
        po = (to * mf).sum(1) / mf.sum(1).clamp(min=1)
        te = torch.cat([self.ie(ti).squeeze(1), self.ce(tc).squeeze(1)], -1)
        return self.mlp(torch.cat([po, te], -1)), None

# ===== Training =====
def train_epoch(model, loader, optimizer, criterion):
    model.train(); tl = 0; n = 0
    for b in loader:
        optimizer.zero_grad()
        logits, _ = model(b['hist_items'].to(device), b['hist_cats'].to(device),
                          b['hist_len'].to(device), b['target_item'].to(device), b['target_cat'].to(device))
        loss = criterion(logits, b['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); tl += loss.item(); n += 1
    return tl / n

def evaluate_model(model, loader, criterion):
    model.eval(); tl = 0; n = 0; labels = []; preds = []; slens = []
    with torch.no_grad():
        for b in loader:
            logits, _ = model(b['hist_items'].to(device), b['hist_cats'].to(device),
                              b['hist_len'].to(device), b['target_item'].to(device), b['target_cat'].to(device))
            loss = criterion(logits, b['label'].to(device))
            tl += loss.item(); n += 1
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(b['label'].numpy().flatten())
            slens.extend(b['hist_len'].numpy().flatten())
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, np.clip(preds, 1e-7, 1-1e-7))
    return tl/n, auc, ll, np.array(labels), np.array(preds), np.array(slens)

# ===== Train all models =====
criterion = nn.BCEWithLogitsLoss()
model_configs = {
    'Mean Pooling': lambda: MeanPoolingModel(n_items, n_categories),
    'DIN': lambda: DIN(n_items, n_categories),
    'DIEN (GRU+Attn)': lambda: DIEN(n_items, n_categories),
    'BST': lambda: BST(n_items, n_categories, ms=MAX_SEQ_LEN),
}

results = {}
for name, fn in model_configs.items():
    print(f'\n{"="*60}\nTraining: {name}\n{"="*60}')
    model = fn().to(device)
    np_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=1, factor=0.5)
    best_auc = 0; pc = 0; times = []
    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tl = train_epoch(model, train_loader, opt, criterion)
        et = time.time() - t0; times.append(et)
        _, auc, ll, labels, preds, slens = evaluate_model(model, test_loader, criterion)
        sch.step(auc)
        imp = ''
        if auc > best_auc:
            best_auc = auc; best_ll = ll; best_labels = labels; best_preds = preds; best_slens = slens
            pc = 0; imp = ' *'
            torch.save(model.state_dict(), os.path.join(PROCESSED_DIR, f'{name.lower().replace(" ","_")}_best.pt'))
        else: pc += 1
        print(f'  Epoch {ep+1:2d} | Loss: {tl:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f} | Time: {et:.1f}s{imp}')
        if pc >= PATIENCE: print('  Early stopping.'); break
    results[name] = {'auc': best_auc, 'logloss': best_ll, 'n_params': np_count,
                     'avg_epoch_time': np.mean(times), 'n_epochs': len(times)}
    print(f'  Best AUC: {best_auc:.4f}')

# ===== Summary =====
print('\n' + '='*80)
print('                  MODEL COMPARISON RESULTS')
print('='*80)
for name in sorted(results, key=lambda x: results[x]['auc'], reverse=True):
    r = results[name]
    print(f"{name:20s} | AUC: {r['auc']:.4f} | LogLoss: {r['logloss']:.4f} | "
          f"Params: {r['n_params']:>10,} | {r['avg_epoch_time']:.1f}s/epoch")

# Save
rp = os.path.join(PROCESSED_DIR, 'taobao_comparison_results.json')
save_data = {k: {kk: vv for kk, vv in v.items() if kk not in ('labels', 'preds', 'seq_lens')}
             for k, v in results.items()}
with open(rp, 'w') as f: json.dump(save_data, f, indent=2)
print(f'\nResults saved to {rp}')

# ===== Plots =====
sns.set_style('whitegrid')
colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
names = list(results.keys())
aucs = [results[n]['auc'] for n in names]
lls = [results[n]['logloss'] for n in names]
ets = [results[n]['avg_epoch_time'] for n in names]

bars = axes[0].bar(range(len(names)), aucs, color=colors, edgecolor='white')
axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=10)
axes[0].set_ylabel('AUC'); axes[0].set_title('Test AUC (higher is better)')
for b, v in zip(bars, aucs):
    axes[0].text(b.get_x()+b.get_width()/2., b.get_height()+0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
axes[0].set_ylim(min(aucs)-0.03, max(aucs)+0.02)

bars = axes[1].bar(range(len(names)), lls, color=colors, edgecolor='white')
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=10)
axes[1].set_ylabel('LogLoss'); axes[1].set_title('Test LogLoss (lower is better)')
for b, v in zip(bars, lls):
    axes[1].text(b.get_x()+b.get_width()/2., b.get_height()+0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

bars = axes[2].bar(range(len(names)), ets, color=colors, edgecolor='white')
axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=10)
axes[2].set_ylabel('Seconds per Epoch'); axes[2].set_title('Training Efficiency')
for b, v in zip(bars, ets):
    axes[2].text(b.get_x()+b.get_width()/2., b.get_height()+0.5, f'{v:.1f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
print('Saved model_comparison.png')

print('\nALL DONE!')
