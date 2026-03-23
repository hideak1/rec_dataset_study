"""Improve EDA notebooks with better explanations:
1. Column meanings for each dataset
2. 'Why' explanations for each analysis step
3. Feature correlation analysis section
"""
import json
import os

def load_nb(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source if isinstance(source, list) else [source], "outputs": [], "execution_count": None}

BASE = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

# ============================================================
# NOTEBOOK 1: Criteo EDA
# ============================================================
print("Updating Criteo EDA...")
nb1_path = os.path.join(BASE, '01_criteo_feature_crossing', '01_data_exploration.ipynb')
nb1 = load_nb(nb1_path)

# Find the first code cell and insert dataset description before it
for i, cell in enumerate(nb1['cells']):
    if cell['cell_type'] == 'markdown' and 'Table of Contents' in ''.join(cell['source']):
        # Insert after table of contents
        nb1['cells'].insert(i + 1, md("""## Dataset Background & Task Definition

> **Why CTR Prediction?**
>
> Click-Through Rate (CTR) prediction is the backbone of online advertising. When a user
> visits a webpage, an ad auction determines which ads to show. The system must predict
> the probability that the user will click each candidate ad. Accurate CTR prediction
> directly impacts:
> - **Revenue**: Advertisers pay per click (CPC model), so accurate CTR = accurate revenue estimation
> - **User experience**: Showing relevant ads reduces ad fatigue
> - **Advertiser ROI**: Better predictions mean better budget allocation

### Column Descriptions

The Criteo dataset contains **anonymized** ad click logs. Each row is one ad impression:

| Column | Name | Type | Description |
|--------|------|------|-------------|
| Label | — | Binary (0/1) | 1 = user clicked the ad, 0 = not clicked |
| I1-I13 | Numerical features | Float | 13 anonymized numerical features (e.g., ad position, page depth, historical CTR). Values are real-valued and may contain missing entries. |
| C1-C26 | Categorical features | String (hashed) | 26 anonymized categorical features (e.g., ad category, device type, publisher domain, user segment). Values are 32-bit hashed strings. |

> **Why are features anonymized?**
>
> Criteo anonymizes features to protect advertiser privacy and prevent reverse-engineering
> of their ad serving logic. While this limits interpretability, it makes the dataset
> safe for public research. The anonymization does NOT affect model training — models
> learn feature interactions regardless of semantic meaning.

### What We Will Analyze and Why

| Analysis | Why It Matters |
|----------|---------------|
| **Class balance** | Imbalanced CTR (~26%) affects loss function choice and threshold tuning |
| **Missing values** | Many features have missing values — we need imputation strategies before modeling |
| **Feature distributions** | Skewed distributions benefit from log transforms; helps choose embedding dims |
| **Feature cardinality** | High-cardinality categoricals need embeddings (not one-hot); determines embedding table sizes |
| **Feature correlations** | Redundant features waste parameters; correlated features inform feature crossing strategy |"""))
        break

# Find a good spot to add correlation analysis
for i, cell in enumerate(nb1['cells']):
    src = ''.join(cell['source']) if 'source' in cell else ''
    if 'Key Takeaways' in src or 'Summary' in src:
        # Insert correlation analysis before summary
        nb1['cells'].insert(i, code("""# Feature Correlation Analysis
# Why: Correlated features indicate redundancy (waste of parameters) and
# suggest which features might benefit from explicit crossing.
# How: We compute pairwise Pearson correlation between numerical features.

if 'train_df' in dir():
    num_cols = [c for c in train_df.columns if c.startswith('I')]
    corr_matrix = train_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Numerical Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Find highly correlated pairs
    high_corr = []
    for j in range(len(num_cols)):
        for k in range(j+1, len(num_cols)):
            r = abs(corr_matrix.iloc[j, k])
            if r > 0.5:
                high_corr.append((num_cols[j], num_cols[k], corr_matrix.iloc[j, k]))

    if high_corr:
        print('Highly correlated feature pairs (|r| > 0.5):')
        for f1, f2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f'  {f1} <-> {f2}: r = {r:.3f}')
        print('\\nThese pairs are candidates for feature crossing in DeepFM/DCN.')
    else:
        print('No highly correlated pairs found — features are relatively independent.')"""))

        nb1['cells'].insert(i, md("""## Feature Correlation Analysis

> **Why analyze feature correlations?**
>
> Feature correlation analysis serves two purposes in CTR prediction:
>
> 1. **Identify redundancy**: Highly correlated features (|r| > 0.8) carry similar
>    information. Including both wastes embedding parameters without adding signal.
>
> 2. **Guide feature crossing**: Moderately correlated features (0.3 < |r| < 0.7)
>    are prime candidates for explicit feature crossing — their interaction captures
>    non-linear patterns that individual features miss. This is exactly what DeepFM's
>    FM component and DCN's cross layers aim to learn.
>
> **How it works**: Pearson correlation measures linear association between two variables.
> Values range from -1 (perfect negative) to +1 (perfect positive). We compute this
> pairwise for all numerical features and visualize as a heatmap."""))
        break

save_nb(nb1, nb1_path)
print("  Done")


# ============================================================
# NOTEBOOK 2: Taobao EDA
# ============================================================
print("Updating Taobao EDA...")
nb2_path = os.path.join(BASE, '02_taobao_sequential', '01_data_exploration.ipynb')
nb2 = load_nb(nb2_path)

# Find table of contents and add analysis rationale after it
for i, cell in enumerate(nb2['cells']):
    if cell['cell_type'] == 'markdown' and 'Table of Contents' in ''.join(cell['source']):
        nb2['cells'].insert(i + 1, md("""### What We Will Analyze and Why

| Analysis | Why It Matters |
|----------|---------------|
| **Behavior distribution** | The ratio of PV:Cart:Fav:Buy tells us how selective users are — low conversion means we need implicit signals |
| **Temporal patterns** | Peak hours and weekly cycles determine when user intent is strongest — models should capture these patterns |
| **User engagement** | Power-law distribution means a few users have many interactions while most have few — affects training batch composition |
| **Item popularity** | Long-tail distribution means most items are rarely interacted with — cold-start problem for new items |
| **Sequence lengths** | Determines max_seq_len hyperparameter for DIN/BST/SASRec — too short loses context, too long wastes memory |
| **Category diversity** | Users who browse many categories have different intent patterns than those who stick to one |"""))
        break

save_nb(nb2, nb2_path)
print("  Done")


# ============================================================
# NOTEBOOK 3: Tenrec EDA
# ============================================================
print("Updating Tenrec EDA...")
nb3_path = os.path.join(BASE, '03_tenrec_cvr', '01_data_exploration.ipynb')
nb3 = load_nb(nb3_path)

# Add column meanings after the first markdown cell
for i, cell in enumerate(nb3['cells']):
    if cell['cell_type'] == 'markdown' and 'Table of Contents' in ''.join(cell['source']):
        nb3['cells'].insert(i + 1, md("""### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | Integer | Unique user identifier |
| `item_id` | Integer | Unique video/content identifier |
| `click` | Binary (0/1) | Whether user clicked on the content |
| `follow` | Binary (0/1) | Whether user followed the content creator |
| `like` | Binary (0/1) | Whether user liked the content (**our CVR target**) |
| `share` | Binary (0/1) | Whether user shared the content |
| `video_category` | Integer | Category of the video content |
| `watching_times` | Float | How long user watched the video |
| `gender` | Integer | User gender (demographic feature) |
| `age` | Integer | User age group (demographic feature) |
| `hist_1` to `hist_10` | Integer | User's 10 most recent item interactions (behavioral history) |

### Task Definition

| Task | Label | Training Space | Model |
|------|-------|---------------|-------|
| **CTR** | `click` | All impressions | Shared with CVR |
| **CVR** (Like) | `like` | Only clicked items (biased!) | Naive approach |
| **CTCVR** | `click × like` | All impressions (entire space) | ESMM approach |

> **Why use "like" as CVR proxy?**
>
> In Tenrec, there is no purchase signal. We use `like` as a proxy for conversion
> because it represents a deeper engagement than just clicking — the user actively
> expressed positive sentiment. The click→like funnel mirrors the impression→click→purchase
> funnel in e-commerce.

### What We Will Analyze and Why

| Analysis | Why It Matters |
|----------|---------------|
| **Label distributions** | Extreme sparsity in like/share signals determines class weighting and evaluation metrics |
| **Conversion funnel** | Click→Like ratio quantifies the SSB problem that ESMM addresses |
| **User demographics** | Gender/age segments may have different conversion rates — multi-task models should handle this |
| **Item categories** | Some categories convert better — per-category bias analysis motivates entire-space training |
| **Behavior history** | Sparsity in hist_1-10 features affects how much sequential context is available |"""))
        break

save_nb(nb3, nb3_path)
print("  Done")


# ============================================================
# NOTEBOOK 4: Ali-CCP EDA
# ============================================================
print("Updating Ali-CCP EDA...")
nb4_path = os.path.join(BASE, '04_aliccp_cvr', '01_data_exploration.ipynb')
nb4 = load_nb(nb4_path)

# This notebook is the most bare (only 6 markdown cells). Add comprehensive content.
# Find the data format section and insert detailed column descriptions after it
for i, cell in enumerate(nb4['cells']):
    src = ''.join(cell.get('source', []))
    if 'Data Loading' in src and cell['cell_type'] == 'markdown':
        nb4['cells'].insert(i + 1, md("""### Column Descriptions

The Ali-CCP dataset (released with the ESMM paper, SIGIR 2018) contains real ad impression
logs from Taobao. Features are organized into **fields** (feature groups):

#### Sample Labels (in sample_skeleton files)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | Integer | Unique impression identifier |
| `click` | Binary (0/1) | Whether user clicked the ad |
| `conversion` | Binary (0/1) | Whether user purchased after clicking (our CVR target) |
| `hash_id` | String | Links to shared user/context features in common_features |

#### Feature Fields

| Field ID | Category | Description | Cardinality | Values/Sample |
|----------|----------|-------------|-------------|---------------|
| **User Features** | | | | |
| 101 | User ID | Unique user identifier | ~92K | 1 |
| 121 | User Profile | User profile segment | ~97 | 1 |
| 122 | User Group | User marketing group | ~13 | 1 |
| 124 | Gender | User gender | 2 | 1 |
| 125 | Age | User age bracket | 7 | 1 |
| 126 | Consumption Level | User spending tier | 3 | 1 |
| 127 | User Feature | Additional user segment | 3 | 1 |
| 128 | Occupation | User occupation status | 2 | 1 |
| 129 | Geography | User geographic region | 4 | 1 |
| **Item Features** | | | | |
| 205 | Item ID | Unique item/ad identifier | ~970K | 1 |
| 206 | Item Category | Product category | ~7K | 1 |
| 207 | Item Feature | Additional item attribute | ~348K | 1 |
| 210 | Item Intention | Ad targeting intent | ~95K | 1 |
| 216 | Item Brand | Product brand | ~132K | 1 |
| 301 | Context | Contextual feature (e.g., ad slot) | 3 | 1 |
| **User Behavioral History** (multi-valued) | | | | |
| 109_14 | User Categories | Categories user has browsed | ~11K | avg ~60 |
| 110_14 | User Shops | Shops user has visited | ~1.5M | avg ~207 |
| 127_14 | User Brands | Brands user has interacted with | ~307K | avg ~127 |
| 150_14 | User Intentions | User inferred purchase intents | ~97K | avg ~128 |
| **Cross Features** | | | | |
| 508 | User-Item Categories | Cross between user browsing and item category | ~6K | 1 |
| 509 | User-Item Shops | Cross between user shop history and item shop | ~183K | 1 |
| 702 | User-Item Cross | User-item combination feature | ~79K | 1 |
| 853 | Cross Feature | Additional combination feature | ~69K | 1 |

> **Why are user behavioral history fields multi-valued?**
>
> Fields like 110_14 (user shops) contain the user's entire browsing/purchase history
> across shops, encoded as multiple (feature_id, value) pairs. The `value` is typically
> log(count+1) representing engagement intensity. A user who visited 200 shops has 200
> values in this field. This rich behavioral signal is what makes DIN/BST-style attention
> models effective — but naive encoding (taking only the first value) loses 99% of this signal.

### Task Definition

**Primary Task: Post-Click Conversion Rate (CVR) Prediction**

The Ali-CCP dataset was released alongside the ESMM paper to benchmark solutions for
the **Sample Selection Bias** problem in CVR prediction:

| Metric | Definition | Positive Rate | Challenge |
|--------|-----------|--------------|-----------|
| **CTR** | P(click \\| impression) | ~4.6% | Standard classification |
| **CVR** | P(conversion \\| click) | ~0.6% | Only observable for clicked items (SSB!) |
| **CTCVR** | P(conversion \\| impression) = CTR × CVR | ~0.03% | Extreme sparsity |

> **Why is 0.03% CTCVR AUC of 0.62 actually meaningful?**
>
> With only 0.03% positives, random prediction gives AUC = 0.50. An AUC of 0.62 means
> the model ranks actual converters above 62% of non-converters on average — a significant
> commercial advantage when applied to billions of daily impressions.

### What We Will Analyze and Why

| Analysis | Why It Matters |
|----------|---------------|
| **Conversion funnel** | Quantifies the extreme imbalance that makes CVR prediction hard |
| **Feature field statistics** | Identifies which fields carry the most signal (multi-valued behavioral history) |
| **SSB visualization** | Demonstrates why naive CVR training is biased and motivates ESMM |
| **Feature coverage** | Measures how many fields are populated per sample — sparse features need special handling |"""))
        break

save_nb(nb4, nb4_path)
print("  Done")

# Also copy updated notebooks to docs
import shutil
for subdir in ['01_criteo_feature_crossing', '02_taobao_sequential', '03_tenrec_cvr', '04_aliccp_cvr']:
    src_dir = os.path.join(BASE, subdir)
    dst_dir = os.path.join(BASE, '..', 'docs', 'notebooks', subdir)
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.endswith('.ipynb'):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print("\nAll EDA notebooks updated and copied to docs/")
