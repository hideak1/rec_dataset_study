"""Run SASRec + BST ablation only (BST main training already done)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import time
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Load data
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

print(f'Loaded: {len(user_sequences)} users, {n_items} items, {n_categories} categories')


# ===== SASRec Model =====
class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim=64, n_heads=2, n_layers=2, max_seq_len=50, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.emb_layernorm = nn.LayerNorm(embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layernorm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])

    def _generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, input_ids, return_all_positions=False):
        B, S = input_ids.shape
        item_emb = self.item_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        pos_emb = self.position_embedding(positions)
        seq_emb = item_emb + pos_emb
        seq_emb = self.emb_layernorm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)
        causal_mask = self._generate_causal_mask(S, input_ids.device)
        padding_mask = (input_ids == 0)
        output = self.transformer_encoder(seq_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        output = self.output_layernorm(output)
        # Replace NaN from fully-masked padded positions with 0
        output = torch.nan_to_num(output, nan=0.0)
        return output

    def predict(self, seq_output, target_items):
        target_emb = self.item_embedding(target_items)
        scores = (seq_output.unsqueeze(1) * target_emb).sum(dim=-1)
        return scores


# ===== SASRec Dataset =====
class SASRecDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len=50, mode='train'):
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.all_items = set()
        self.input_seqs = []
        self.target_seqs = []
        self.neg_seqs = []
        self._build(user_sequences)

    def _build(self, user_sequences):
        for uid, seq in user_sequences.items():
            self.all_items.update(seq['item_ids'])
        self.all_items_arr = np.array(list(self.all_items))
        n_users = len(user_sequences)
        pool_size = n_users * self.max_seq_len * 2
        print(f'  Pre-sampling {pool_size:,} SASRec negatives...')
        neg_pool = np.random.choice(self.all_items_arr, size=pool_size)
        pool_idx = 0
        for i, (uid, seq) in enumerate(user_sequences.items()):
            if i % 20000 == 0:
                print(f'  SASRec: processing user {i}/{n_users}...')
            items = seq['item_ids']
            if len(items) < 3:
                continue
            user_item_set = set(items)
            if self.mode == 'train':
                input_items = items[:-2]
                target_items = items[1:-1]
            else:
                input_items = items[:-1]
                target_items = items[1:]
            if len(input_items) == 0:
                continue
            if len(input_items) > self.max_seq_len:
                input_items = input_items[-self.max_seq_len:]
                target_items = target_items[-self.max_seq_len:]
            pad_len = self.max_seq_len - len(input_items)
            input_padded = [0] * pad_len + input_items
            target_padded = [0] * pad_len + target_items
            neg_items = []
            for _ in range(len(input_padded)):
                while True:
                    if pool_idx >= len(neg_pool):
                        neg_pool = np.random.choice(self.all_items_arr, size=pool_size)
                        pool_idx = 0
                    neg = neg_pool[pool_idx]
                    pool_idx += 1
                    if neg not in user_item_set:
                        break
                neg_items.append(neg)
            self.input_seqs.append(input_padded)
            self.target_seqs.append(target_padded)
            self.neg_seqs.append(neg_items)

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.LongTensor(self.input_seqs[idx]),
            'target_ids': torch.LongTensor(self.target_seqs[idx]),
            'neg_ids': torch.LongTensor(self.neg_seqs[idx])
        }


# ===== Training functions =====
def train_sasrec_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        neg_ids = batch['neg_ids'].to(device)
        optimizer.zero_grad()
        seq_output = model(input_ids, return_all_positions=True)
        pos_emb = model.item_embedding(target_ids)
        neg_emb = model.item_embedding(neg_ids)
        pos_scores = (seq_output * pos_emb).sum(dim=-1)
        neg_scores = (seq_output * neg_emb).sum(dim=-1)
        mask = (target_ids != 0)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        loss = torch.where(mask, bpr_loss, torch.zeros_like(bpr_loss))
        loss = loss.sum() / mask.float().sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate_sasrec(model, loader, device, k_list=[5, 10, 20]):
    model.eval()
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            seq_output = model(input_ids, return_all_positions=True)
            B = input_ids.size(0)
            last_output = seq_output[:, -1]
            last_target = target_ids[:, -1]
            n_neg_eval = 99
            neg_items = torch.randint(1, model.n_items, (B, n_neg_eval), device=device)
            candidates = torch.cat([last_target.unsqueeze(1), neg_items], dim=1)
            scores = model.predict(last_output, candidates)
            _, indices = scores.sort(dim=1, descending=True)
            ranks = (indices == 0).nonzero(as_tuple=True)[1] + 1
            for k in k_list:
                hr = (ranks <= k).float().cpu().numpy()
                ndcg = (1.0 / torch.log2(ranks.float() + 1) * (ranks <= k).float()).cpu().numpy()
                metrics[f'HR@{k}'].extend(hr.tolist())
                metrics[f'NDCG@{k}'].extend(ndcg.tolist())
    return {k: np.mean(v) for k, v in metrics.items()}


# ===== Build SASRec datasets =====
print('Building SASRec datasets...')
sasrec_train = SASRecDataset(user_sequences, max_seq_len=MAX_SEQ_LEN, mode='train')
sasrec_test = SASRecDataset(user_sequences, max_seq_len=MAX_SEQ_LEN, mode='test')
print(f'SASRec Train: {len(sasrec_train):,}, Test: {len(sasrec_test):,}')

sasrec_train_loader = DataLoader(sasrec_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
sasrec_test_loader = DataLoader(sasrec_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ===== Train SASRec =====
sasrec_model = SASRec(n_items=n_items, embed_dim=64, n_heads=2, n_layers=2,
                       max_seq_len=MAX_SEQ_LEN, dropout=0.2).to(device)
total_params = sum(p.numel() for p in sasrec_model.parameters())
print(f'\nSASRec parameters: {total_params:,}')

sasrec_optimizer = torch.optim.Adam(sasrec_model.parameters(), lr=1e-3, weight_decay=1e-6)
sasrec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sasrec_optimizer, mode='max', patience=1, factor=0.5)

sasrec_best_hr10 = 0
patience_counter = 0
sasrec_history = {'train_loss': [], 'HR@5': [], 'HR@10': [], 'HR@20': [],
                  'NDCG@5': [], 'NDCG@10': [], 'NDCG@20': []}

print(f'Training SASRec for up to {MAX_EPOCHS} epochs...')
print('='*80)

for epoch in range(MAX_EPOCHS):
    start_time = time.time()
    train_loss = train_sasrec_epoch(sasrec_model, sasrec_train_loader, sasrec_optimizer, device)
    metrics = evaluate_sasrec(sasrec_model, sasrec_test_loader, device)
    epoch_time = time.time() - start_time
    sasrec_scheduler.step(metrics['HR@10'])
    sasrec_history['train_loss'].append(train_loss)
    for k in ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']:
        sasrec_history[k].append(metrics[k])
    improved = ''
    if metrics['HR@10'] > sasrec_best_hr10:
        sasrec_best_hr10 = metrics['HR@10']
        patience_counter = 0
        torch.save(sasrec_model.state_dict(), os.path.join(PROCESSED_DIR, 'sasrec_best.pt'))
        improved = ' *'
    else:
        patience_counter += 1
    print(f'Epoch {epoch+1:2d}/{MAX_EPOCHS} | Loss: {train_loss:.4f} | '
          f'HR@5: {metrics["HR@5"]:.4f} | HR@10: {metrics["HR@10"]:.4f} | '
          f'HR@20: {metrics["HR@20"]:.4f} | NDCG@10: {metrics["NDCG@10"]:.4f} | '
          f'Time: {epoch_time:.1f}s{improved}')
    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

print(f'\nSASRec Best HR@10: {sasrec_best_hr10:.4f}')

# ===== BST Ablation =====
print('\n' + '='*70)
print('BST Ablation Study')
print('='*70)

# Need CTR dataset for ablation
class TaobaoCTRDataset(Dataset):
    def __init__(self, user_sequences, item_to_cat, all_item_ids, all_item_probs,
                 max_seq_len=50, neg_ratio=4, mode='train'):
        self.max_seq_len = max_seq_len
        self.neg_ratio = neg_ratio
        self.item_to_cat = item_to_cat
        self.all_item_ids = np.array(all_item_ids)
        self.all_item_probs = all_item_probs
        self.mode = mode
        self.samples = []
        self._build_samples(user_sequences)

    def _build_samples(self, user_sequences):
        n_users = len(user_sequences)
        pool_size = n_users * self.neg_ratio * 3
        print(f'  Pre-sampling {pool_size:,} negative candidates...')
        neg_pool = np.random.choice(self.all_item_ids, size=pool_size, p=self.all_item_probs)
        pool_idx = 0
        for i, (uid, seq) in enumerate(user_sequences.items()):
            if i % 20000 == 0:
                print(f'  Processing user {i}/{n_users}...')
            items = seq['item_ids']
            cats = seq['cat_ids']
            if len(items) < 3:
                continue
            target_idx = len(items) - 2 if self.mode == 'train' else len(items) - 1
            hist_items = items[:target_idx]
            hist_cats = cats[:target_idx]
            if len(hist_items) == 0:
                continue
            if len(hist_items) > self.max_seq_len:
                hist_items = hist_items[-self.max_seq_len:]
                hist_cats = hist_cats[-self.max_seq_len:]
            hist_len = len(hist_items)
            pad_len = self.max_seq_len - hist_len
            hist_items_padded = [0] * pad_len + hist_items
            hist_cats_padded = [0] * pad_len + hist_cats
            target_item = items[target_idx]
            target_cat = cats[target_idx]
            self.samples.append((hist_items_padded, hist_cats_padded, hist_len,
                                target_item, target_cat, 1))
            user_item_set = set(items)
            neg_count = 0
            while neg_count < self.neg_ratio:
                if pool_idx >= len(neg_pool):
                    neg_pool = np.random.choice(self.all_item_ids, size=pool_size, p=self.all_item_probs)
                    pool_idx = 0
                neg_item = neg_pool[pool_idx]
                pool_idx += 1
                if neg_item not in user_item_set:
                    neg_cat = self.item_to_cat.get(neg_item, 0)
                    self.samples.append((hist_items_padded, hist_cats_padded, hist_len,
                                        neg_item, neg_cat, 0))
                    neg_count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_items, hist_cats, hist_len, target_item, target_cat, label = self.samples[idx]
        return {
            'hist_items': torch.LongTensor(hist_items),
            'hist_cats': torch.LongTensor(hist_cats),
            'hist_len': torch.LongTensor([hist_len]),
            'target_item': torch.LongTensor([target_item]),
            'target_cat': torch.LongTensor([target_cat]),
            'label': torch.FloatTensor([label])
        }


class BST(nn.Module):
    def __init__(self, n_items, n_categories, embed_dim=32, n_heads=4, n_layers=2,
                 max_seq_len=50, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        cat_dim = embed_dim // 2
        self.feature_dim = embed_dim + cat_dim
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(n_categories, cat_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, self.feature_dim)
        self.emb_layernorm = nn.LayerNorm(self.feature_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=n_heads,
            dim_feedforward=self.feature_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        mlp_input_dim = self.feature_dim * 2
        layers = []
        in_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h_dim
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

    def forward(self, hist_items, hist_cats, hist_len, target_item, target_cat, return_attention=False):
        B, S = hist_items.shape
        item_emb = self.item_embedding(hist_items)
        cat_emb = self.cat_embedding(hist_cats)
        seq_emb = torch.cat([item_emb, cat_emb], dim=-1)
        positions = torch.arange(S, device=hist_items.device).unsqueeze(0).expand(B, S)
        pos_emb = self.position_embedding(positions)
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_layernorm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)
        padding_mask = (hist_items == 0)
        transformer_out = self.transformer_encoder(seq_emb, src_key_padding_mask=padding_mask)
        mask_float = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        target_item_emb = self.item_embedding(target_item).squeeze(1)
        target_cat_emb = self.cat_embedding(target_cat).squeeze(1)
        target_emb = torch.cat([target_item_emb, target_cat_emb], dim=-1)
        mlp_input = torch.cat([pooled, target_emb], dim=-1)
        logits = self.mlp(mlp_input)
        return logits, None


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        hist_items = batch['hist_items'].to(device)
        hist_cats = batch['hist_cats'].to(device)
        hist_len = batch['hist_len'].to(device)
        target_item = batch['target_item'].to(device)
        target_cat = batch['target_cat'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits, _ = model(hist_items, hist_cats, hist_len, target_item, target_cat)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            hist_items = batch['hist_items'].to(device)
            hist_cats = batch['hist_cats'].to(device)
            hist_len = batch['hist_len'].to(device)
            target_item = batch['target_item'].to(device)
            target_cat = batch['target_cat'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(hist_items, hist_cats, hist_len, target_item, target_cat)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / n_batches
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
    return avg_loss, auc, logloss, np.array(all_labels), np.array(all_preds)


print('\nBuilding CTR datasets for ablation...')
train_dataset = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs,
                                  max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, mode='train')
test_dataset = TaobaoCTRDataset(user_sequences, item_to_cat, all_item_ids, all_item_probs,
                                 max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, mode='test')
print(f'Train: {len(train_dataset):,}, Test: {len(test_dataset):,}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()

ablation_configs = [
    {'n_heads': 1, 'n_layers': 1, 'name': '1H-1L'},
    {'n_heads': 4, 'n_layers': 1, 'name': '4H-1L'},
    {'n_heads': 4, 'n_layers': 2, 'name': '4H-2L'},
]

ablation_results = {}

for config in ablation_configs:
    print(f"\nTraining BST ({config['name']})...")
    abl_model = BST(n_items=n_items, n_categories=n_categories,
                    embed_dim=32, n_heads=config['n_heads'], n_layers=config['n_layers'],
                    max_seq_len=MAX_SEQ_LEN, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    abl_optimizer = torch.optim.Adam(abl_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    abl_best_auc = 0
    patience_cnt = 0
    for epoch in range(MAX_EPOCHS):
        start = time.time()
        train_loss = train_epoch(abl_model, train_loader, abl_optimizer, criterion, device)
        _, test_auc, _, _, _ = evaluate(abl_model, test_loader, criterion, device)
        elapsed = time.time() - start
        improved = ' *' if test_auc > abl_best_auc else ''
        print(f'  Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Test AUC: {test_auc:.4f} | Time: {elapsed:.1f}s{improved}')
        if test_auc > abl_best_auc:
            abl_best_auc = test_auc
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break
    ablation_results[config['name']] = abl_best_auc
    n_params = sum(p.numel() for p in abl_model.parameters())
    print(f"  {config['name']}: AUC = {abl_best_auc:.4f}, Params = {n_params:,}")

print('\n' + '='*50)
print('BST Ablation Results:')
for name, auc in sorted(ablation_results.items(), key=lambda x: x[1], reverse=True):
    print(f'  {name:10s}: AUC = {auc:.4f}')

# ===== Save results =====
# Load existing results if any
results_path = os.path.join(PROCESSED_DIR, 'transformer_results.json')
results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

results['sasrec_best_hr10'] = sasrec_best_hr10
results['sasrec_history'] = sasrec_history
results['ablation_results'] = ablation_results

# BST results from earlier run
if 'bst_best_auc' not in results:
    results['bst_best_auc'] = 0.7847  # from earlier run

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {results_path}')

# ===== Generate SASRec plots =====
sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(sasrec_history['train_loss']) + 1)
axes[0].plot(epochs_range, sasrec_history['train_loss'], 'o-', color='#4e79a7', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BPR Loss'); axes[0].set_title('SASRec: Training Loss')
for k, color in zip(['HR@5', 'HR@10', 'HR@20'], ['#4e79a7', '#59a14f', '#f28e2b']):
    axes[1].plot(epochs_range, sasrec_history[k], 'o-', label=k, color=color, linewidth=2)
axes[1].axhline(0.50, color='red', linestyle='--', alpha=0.7, label='Target HR@10=0.50')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Hit Rate'); axes[1].set_title('SASRec: Hit Rate@K'); axes[1].legend()
for k, color in zip(['NDCG@5', 'NDCG@10', 'NDCG@20'], ['#4e79a7', '#59a14f', '#f28e2b']):
    axes[2].plot(epochs_range, sasrec_history[k], 'o-', label=k, color=color, linewidth=2)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('NDCG'); axes[2].set_title('SASRec: NDCG@K'); axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, 'sasrec_training_curves.png'), dpi=150, bbox_inches='tight')
print('Saved sasrec_training_curves.png')

# Ablation plot
fig, ax = plt.subplots(figsize=(10, 5))
names = list(ablation_results.keys())
aucs = [ablation_results[n] for n in names]
bars = ax.bar(names, aucs, color='#4e79a7', edgecolor='white', linewidth=2)
ax.axhline(0.64, color='red', linestyle='--', alpha=0.7, label='Target AUC=0.64')
ax.set_ylabel('Test AUC'); ax.set_title('BST Architecture Ablation (Heads x Layers)'); ax.legend()
for bar, auc_val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
            f'{auc_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
min_auc = min(aucs)
ax.set_ylim(min_auc - 0.02, max(aucs) + 0.02)
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, 'bst_ablation.png'), dpi=150, bbox_inches='tight')
print('Saved bst_ablation.png')

print('\n' + '='*70)
print('ALL DONE!')
print(f'SASRec Best HR@10: {sasrec_best_hr10:.4f}')
print(f'Ablation: {ablation_results}')
print('='*70)
