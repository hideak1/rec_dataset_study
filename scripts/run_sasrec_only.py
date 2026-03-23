"""Run SASRec only — with right-padding to fix NaN from causal mask + left-pad."""
import numpy as np
import pickle
import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

# Load data
print('\nLoading data...')
with open(os.path.join(PROCESSED_DIR, 'taobao_sequential_data.pkl'), 'rb') as f:
    data = pickle.load(f)

user_sequences = data['user_sequences']
n_items = data['n_items']

MAX_SEQ_LEN = 50
BATCH_SIZE = 1024
MAX_EPOCHS = 15
PATIENCE = 3

print(f'Loaded: {len(user_sequences)} users, {n_items} items')


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

    def forward(self, input_ids):
        """input_ids: (B, S) RIGHT-padded with 0."""
        B, S = input_ids.shape
        item_emb = self.item_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        pos_emb = self.position_embedding(positions)
        seq_emb = item_emb + pos_emb
        seq_emb = self.emb_layernorm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        # Causal mask: (S, S) True = masked (future positions)
        causal_mask = torch.triu(torch.ones(S, S, device=input_ids.device), diagonal=1).bool()
        # Padding mask: (B, S) True = padded
        padding_mask = (input_ids == 0)

        output = self.transformer_encoder(seq_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        output = self.output_layernorm(output)
        return output

    def predict(self, seq_output, target_items):
        target_emb = self.item_embedding(target_items)
        scores = (seq_output.unsqueeze(1) * target_emb).sum(dim=-1)
        return scores


class SASRecDataset(Dataset):
    """RIGHT-padded SASRec dataset for next-item prediction."""
    def __init__(self, user_sequences, max_seq_len=50, mode='train'):
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.all_items = set()
        self.input_seqs = []
        self.target_seqs = []
        self.neg_seqs = []
        self.seq_lens = []  # actual length of each sequence
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
            seq_len = len(input_items)
            # RIGHT-pad (items first, then zeros)
            pad_len = self.max_seq_len - seq_len
            input_padded = input_items + [0] * pad_len
            target_padded = target_items + [0] * pad_len
            neg_items = []
            for _ in range(self.max_seq_len):
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
            self.seq_lens.append(seq_len)

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.LongTensor(self.input_seqs[idx]),
            'target_ids': torch.LongTensor(self.target_seqs[idx]),
            'neg_ids': torch.LongTensor(self.neg_seqs[idx]),
            'seq_len': torch.LongTensor([self.seq_lens[idx]])
        }


def train_sasrec_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        neg_ids = batch['neg_ids'].to(device)
        optimizer.zero_grad()
        seq_output = model(input_ids)
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
            seq_lens = batch['seq_len'].to(device).squeeze(-1)
            B = input_ids.size(0)
            seq_output = model(input_ids)
            # Get representation at LAST VALID position (right-padded: seq_len - 1)
            last_pos = seq_lens - 1  # (B,)
            last_output = seq_output[torch.arange(B, device=device), last_pos]  # (B, D)
            # Target at last valid position
            last_target = target_ids[torch.arange(B, device=device), last_pos]  # (B,)
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


# Build datasets
print('Building SASRec datasets (right-padded)...')
sasrec_train = SASRecDataset(user_sequences, max_seq_len=MAX_SEQ_LEN, mode='train')
sasrec_test = SASRecDataset(user_sequences, max_seq_len=MAX_SEQ_LEN, mode='test')
print(f'SASRec Train: {len(sasrec_train):,}, Test: {len(sasrec_test):,}')

train_loader = DataLoader(sasrec_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(sasrec_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Train
sasrec_model = SASRec(n_items=n_items, embed_dim=64, n_heads=2, n_layers=2,
                       max_seq_len=MAX_SEQ_LEN, dropout=0.2).to(device)
print(f'\nSASRec parameters: {sum(p.numel() for p in sasrec_model.parameters()):,}')

optimizer = torch.optim.Adam(sasrec_model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)

best_hr10 = 0
patience_counter = 0
history = {'train_loss': [], 'HR@5': [], 'HR@10': [], 'HR@20': [],
           'NDCG@5': [], 'NDCG@10': [], 'NDCG@20': []}

print(f'Training SASRec for up to {MAX_EPOCHS} epochs...')
print('='*80)

for epoch in range(MAX_EPOCHS):
    start = time.time()
    train_loss = train_sasrec_epoch(sasrec_model, train_loader, optimizer, device)
    metrics = evaluate_sasrec(sasrec_model, test_loader, device)
    elapsed = time.time() - start
    scheduler.step(metrics['HR@10'])
    history['train_loss'].append(train_loss)
    for k in ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']:
        history[k].append(metrics[k])
    improved = ''
    if metrics['HR@10'] > best_hr10:
        best_hr10 = metrics['HR@10']
        patience_counter = 0
        torch.save(sasrec_model.state_dict(), os.path.join(PROCESSED_DIR, 'sasrec_best.pt'))
        improved = ' *'
    else:
        patience_counter += 1
    print(f'Epoch {epoch+1:2d}/{MAX_EPOCHS} | Loss: {train_loss:.4f} | '
          f'HR@5: {metrics["HR@5"]:.4f} | HR@10: {metrics["HR@10"]:.4f} | '
          f'HR@20: {metrics["HR@20"]:.4f} | NDCG@10: {metrics["NDCG@10"]:.4f} | '
          f'Time: {elapsed:.1f}s{improved}')
    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

print(f'\nSASRec Best HR@10: {best_hr10:.4f}')

# Save results
results_path = os.path.join(PROCESSED_DIR, 'transformer_results.json')
results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
results['sasrec_best_hr10'] = best_hr10
results['sasrec_history'] = history
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {results_path}')

print('\nDONE!')
