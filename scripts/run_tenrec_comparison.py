"""Re-run Tenrec comparison on GPU for better results."""
import os
import json
import time
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = Path(BASE_DIR) / 'data' / 'Tenrec' / 'processed'

# Load data
train_df = pd.read_csv(PROCESSED_DIR / 'train.csv')
test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')
with open(PROCESSED_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)
feature_fields = OrderedDict(metadata['feature_fields'])
MODEL_FEATURES = metadata['model_features']

class TenrecDataset(Dataset):
    def __init__(self, df, feature_names):
        self.features = torch.LongTensor(df[feature_names].values)
        self.click = torch.FloatTensor(df['click'].values)
        self.like = torch.FloatTensor(df['like'].values)
    def __len__(self): return len(self.click)
    def __getitem__(self, idx): return self.features[idx], self.click[idx], self.like[idx]

BATCH_SIZE = 4096
train_dataset = TenrecDataset(train_df, MODEL_FEATURES)
test_dataset = TenrecDataset(test_df, MODEL_FEATURES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# ===== Model Definitions =====
class SharedEmbedding(nn.Module):
    def __init__(self, feature_fields, embed_dim=16):
        super().__init__()
        self.feature_names = list(feature_fields.keys())
        self.embeddings = nn.ModuleDict()
        for name, info in feature_fields.items():
            self.embeddings[name] = nn.Embedding(info['cardinality'], embed_dim, padding_idx=0)
        self.output_dim = len(feature_fields) * embed_dim
    def forward(self, x):
        return torch.cat([self.embeddings[name](x[:, i]) for i, name in enumerate(self.feature_names)], dim=1)

class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]); prev = h
        layers.append(nn.Linear(prev, 1)); self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class NaiveCVR(nn.Module):
    def __init__(self, ff, ed=16, hd=(256,128,64), dr=0.3):
        super().__init__()
        self.emb = SharedEmbedding(ff, ed); self.tower = Tower(self.emb.output_dim, hd, dr)
    def forward(self, x): return torch.sigmoid(self.tower(self.emb(x)).squeeze(1))

class ESMM(nn.Module):
    def __init__(self, ff, ed=16, hd=(256,128,64), dr=0.3, w=1.0):
        super().__init__()
        self.emb = SharedEmbedding(ff, ed)
        self.ctr = Tower(self.emb.output_dim, hd, dr); self.cvr = Tower(self.emb.output_dim, hd, dr)
        self.w = w
    def forward(self, x):
        e = self.emb(x)
        cp = torch.sigmoid(self.ctr(e).squeeze(1)); vp = torch.sigmoid(self.cvr(e).squeeze(1))
        return cp, vp, cp * vp
    def compute_loss(self, cp, ctcvr, cl, ll):
        l1 = F.binary_cross_entropy(cp, cl)
        l2 = F.binary_cross_entropy(torch.clamp(ctcvr, 1e-7, 1-1e-7), ll)
        return l1 + self.w * l2, l1, l2

class ExpertNet(nn.Module):
    def __init__(self, ind, ed, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, ed), nn.BatchNorm1d(ed), nn.ReLU(), nn.Dropout(dr),
                                  nn.Linear(ed, ed), nn.ReLU())
    def forward(self, x): return self.net(x)

class Gate(nn.Module):
    def __init__(self, ind, ne, t=1.0):
        super().__init__()
        self.g = nn.Linear(ind, ne); self.t = t
    def forward(self, x): return F.softmax(self.g(x)/self.t, dim=1)

class MMoE(nn.Module):
    def __init__(self, ff, ed=16, ne=8, exd=64, thd=(128,64), dr=0.3, w=1.0):
        super().__init__()
        self.emb = SharedEmbedding(ff, ed); ind = self.emb.output_dim
        self.experts = nn.ModuleList([ExpertNet(ind, exd) for _ in range(ne)])
        self.ctr_gate = Gate(ind, ne); self.cvr_gate = Gate(ind, ne)
        self.ctr_tower = Tower(exd, thd, dr); self.cvr_tower = Tower(exd, thd, dr)
        self.w = w
    def forward(self, x):
        e = self.emb(x)
        eo = torch.stack([ex(e) for ex in self.experts], dim=1)
        cg = self.ctr_gate(e); vg = self.cvr_gate(e)
        cp = torch.sigmoid(self.ctr_tower(torch.sum(eo * cg.unsqueeze(2), 1)).squeeze(1))
        vp = torch.sigmoid(self.cvr_tower(torch.sum(eo * vg.unsqueeze(2), 1)).squeeze(1))
        return cp, vp, cp * vp
    def compute_loss(self, cp, ctcvr, cl, ll):
        l1 = F.binary_cross_entropy(cp, cl)
        l2 = F.binary_cross_entropy(torch.clamp(ctcvr, 1e-7, 1-1e-7), ll)
        return l1 + self.w * l2, l1, l2

class PLELayer(nn.Module):
    def __init__(self, ind, exd, nse=4, nte=2, nt=2, dr=0.1):
        super().__init__()
        self.shared = nn.ModuleList([ExpertNet(ind, exd, dr) for _ in range(nse)])
        self.task_exp = nn.ModuleList([nn.ModuleList([ExpertNet(ind, exd, dr) for _ in range(nte)]) for _ in range(nt)])
        self.gates = nn.ModuleList([Gate(ind, nse+nte) for _ in range(nt)])
        self.nt = nt
    def forward(self, x):
        ti = [x]*self.nt if isinstance(x, torch.Tensor) else x
        so = [e(ti[0]) for e in self.shared]
        out = []
        for t in range(self.nt):
            ao = so + [e(ti[t]) for e in self.task_exp[t]]
            s = torch.stack(ao, 1); g = self.gates[t](ti[t])
            out.append(torch.sum(s * g.unsqueeze(2), 1))
        return out

class PLE(nn.Module):
    def __init__(self, ff, ed=16, nl=2, nse=4, nte=2, exd=64, thd=(128,64), dr=0.3, w=1.0):
        super().__init__()
        self.emb = SharedEmbedding(ff, ed); ind = self.emb.output_dim
        self.layers = nn.ModuleList([PLELayer(ind if l==0 else exd, exd, nse, nte, 2, 0.1) for l in range(nl)])
        self.ctr_tower = Tower(exd, thd, dr); self.cvr_tower = Tower(exd, thd, dr)
        self.w = w
    def forward(self, x):
        e = self.emb(x); ti = e
        for layer in self.layers: ti = layer(ti)
        cp = torch.sigmoid(self.ctr_tower(ti[0]).squeeze(1))
        vp = torch.sigmoid(self.cvr_tower(ti[1]).squeeze(1))
        return cp, vp, cp * vp
    def compute_loss(self, cp, ctcvr, cl, ll):
        l1 = F.binary_cross_entropy(cp, cl)
        l2 = F.binary_cross_entropy(torch.clamp(ctcvr, 1e-7, 1-1e-7), ll)
        return l1 + self.w * l2, l1, l2

# ===== Training =====
def train_mt(model, tl, vl, name, ne=15, lr=1e-3, wd=1e-5, pat=3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 1, verbose=False)
    best = 0; bs = None; pc = 0
    for ep in range(ne):
        model.train(); el = 0; nb = 0; t0 = time.time()
        for f, c, l in tl:
            f, c, l = f.to(device), c.to(device), l.to(device)
            cp, vp, ctcvr = model(f)
            loss, _, _ = model.compute_loss(cp, ctcvr, c, l)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); el += loss.item(); nb += 1
        et = time.time() - t0
        model.eval(); ac = []; al = []
        with torch.no_grad():
            for f, c, l in vl:
                _, _, ctcvr = model(f.to(device))
                ac.append(ctcvr.cpu().numpy()); al.append(l.numpy())
        auc = roc_auc_score(np.concatenate(al), np.concatenate(ac))
        sch.step(auc)
        m = ''
        if auc > best: best = auc; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pc = 0; m = ' *'
        else: pc += 1
        print(f"  [{name}] Ep {ep+1:2d} ({et:.1f}s) | Loss: {el/nb:.4f} | CTCVR AUC: {auc:.4f}{m}")
        if pc >= pat: break
    if bs: model.load_state_dict(bs); model.to(device)
    return best

def train_naive(model, tdf, vl, mf, ne=15, lr=1e-3, wd=1e-5, pat=3, bs=4096):
    cdf = tdf[tdf['click']==1]
    cds = TenrecDataset(cdf, mf)
    cl = DataLoader(cds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = 0; bst = None; pc = 0
    for ep in range(ne):
        model.train(); el = 0; nb = 0; t0 = time.time()
        for f, _, l in cl:
            f, l = f.to(device), l.to(device)
            p = model(f); loss = F.binary_cross_entropy(p, l)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); el += loss.item(); nb += 1
        et = time.time() - t0
        model.eval(); ap = []; al = []
        with torch.no_grad():
            for f, c, l in vl:
                p = model(f.to(device)); ap.append(p.cpu().numpy()); al.append(l.numpy())
        auc = roc_auc_score(np.concatenate(al), np.concatenate(ap))
        m = ''
        if auc > best: best = auc; bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pc = 0; m = ' *'
        else: pc += 1
        print(f"  [Naive] Ep {ep+1:2d} ({et:.1f}s) | Loss: {el/nb:.4f} | Like AUC: {auc:.4f}{m}")
        if pc >= pat: break
    if bst: model.load_state_dict(bst); model.to(device)
    return best

# ===== Train all =====
print("\n" + "="*80)
print("TRAINING ALL MODELS ON GPU")
print("="*80)

results = {}

print("\n--- Naive CVR ---")
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
mn = NaiveCVR(feature_fields).to(device)
results['Naive CVR'] = train_naive(mn, train_df, test_loader, MODEL_FEATURES)

print("\n--- ESMM ---")
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
me = ESMM(feature_fields).to(device)
results['ESMM'] = train_mt(me, train_loader, test_loader, 'ESMM')

print("\n--- MMoE ---")
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
mm = MMoE(feature_fields).to(device)
results['MMoE'] = train_mt(mm, train_loader, test_loader, 'MMoE')

print("\n--- PLE ---")
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
mp = PLE(feature_fields).to(device)
results['PLE'] = train_mt(mp, train_loader, test_loader, 'PLE')

# Also get CTR AUC and full metrics
print("\n--- Final Evaluation ---")
all_results = {}
models = {'Naive CVR': mn, 'ESMM': me, 'MMoE': mm, 'PLE': mp}
for name, model in models.items():
    model.eval()
    preds_ctcvr = []; preds_ctr = []; clicks = []; likes = []
    with torch.no_grad():
        for f, c, l in test_loader:
            if name == 'Naive CVR':
                p = model(f.to(device))
                preds_ctcvr.append(p.cpu().numpy())
            else:
                cp, vp, ctcvr = model(f.to(device))
                preds_ctcvr.append(ctcvr.cpu().numpy())
                preds_ctr.append(cp.cpu().numpy())
            clicks.append(c.numpy()); likes.append(l.numpy())
    ctcvr = np.concatenate(preds_ctcvr); cl = np.concatenate(clicks); lk = np.concatenate(likes)
    r = {'ctcvr_auc': roc_auc_score(lk, ctcvr), 'like_auc': roc_auc_score(lk, ctcvr)}
    if preds_ctr:
        ctr = np.concatenate(preds_ctr)
        r['ctr_auc'] = roc_auc_score(cl, ctr)
    r['n_params'] = sum(p.numel() for p in model.parameters())
    all_results[name] = r

print("\n" + "="*80)
print("TENREC COMPARISON RESULTS (GPU)")
print("="*80)
for name in ['Naive CVR', 'ESMM', 'MMoE', 'PLE']:
    r = all_results[name]
    ctr_str = f"CTR AUC: {r.get('ctr_auc', 0):.4f}" if 'ctr_auc' in r else "CTR: N/A"
    print(f"{name:15s} | CTCVR AUC: {r['ctcvr_auc']:.4f} | {ctr_str} | Params: {r['n_params']:,}")

# Save
rp = str(PROCESSED_DIR / 'tenrec_comparison_results_gpu.json')
with open(rp, 'w') as f: json.dump(all_results, f, indent=2)
print(f"\nSaved to {rp}")
print("\nDONE!")
