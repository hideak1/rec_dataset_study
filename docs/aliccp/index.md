# Experiment 3: Tenrec — Multi-Task CVR Prediction

The [Tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec) dataset is a large-scale multi-behavior recommendation dataset from Tencent, containing 120M+ user-item interactions with multiple engagement signals: click, follow, like, and share.

We use **click -> like** as the conversion funnel (analogous to click -> purchase in e-commerce), making this ideal for studying the Sample Selection Bias problem and multi-task CVR models.

## Focus: Multi-Task CVR Prediction

| Notebook | Model | Key Idea |
|----------|-------|----------|
| 3.1 | — | Dataset exploration, multi-task label analysis, SSB problem |
| 3.2 | ESMM | Entire Space Multi-Task: CTCVR = CTR x CVR, eliminates SSB |
| 3.3 | MMoE / PLE | Expert networks with task-specific gating and progressive extraction |
| 3.4 | — | Head-to-head comparison, calibration, business impact |

## Dataset Statistics

- **120M+** user-item interactions
- **Click rate**: ~27%
- **Like rate among clicks**: ~2.1% (conversion analogy)
- **Features**: user_id, item_id, video_category, gender, age, watching_times, behavior history (hist_1-hist_10)

## Target Performance

| Model | Like AUC (all) | CTCVR AUC |
|-------|---------------|-----------|
| NaiveCVR | ~0.65 | ~0.60 |
| ESMM | >= 0.70 | >= 0.64 |
| MMoE | >= 0.71 | >= 0.65 |
| PLE | >= 0.71 | >= 0.65 |
