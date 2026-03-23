# Experiment 4: Ali-CCP — Multi-Task CVR (Industry Benchmark)

## Overview

The **Ali-CCP (Alibaba Click and Conversion Prediction)** dataset is the original
industry benchmark released alongside the ESMM paper (SIGIR 2018). It contains
real-world click and conversion logs from Taobao's advertising system.

## Why Ali-CCP?

- **Original ESMM benchmark**: The dataset that motivated entire-space multi-task learning
- **Extreme sparsity**: ~4.7% CTR, ~0.6% CVR among clicks, ~0.03% CTCVR
- **Production scale**: 42M+ train samples from a real advertising system
- **Rich features**: User demographics, item attributes, and cross features

## Dataset Features

| Field | Description | Type |
|-------|-------------|------|
| 101 | User ID | Sparse |
| 121-129 | User profile (age, gender, geography, etc.) | Sparse |
| 205-216 | Item features (ID, category, brand) | Sparse |
| 109_14, 110_14, 127_14, 150_14 | User behavioral history | Multi-valued |
| 508, 509, 702, 853 | User-item cross features | Sparse |
| 301 | Context feature | Sparse |

## Models

| Model | Key Innovation |
|-------|---------------|
| **ESMM** | CTCVR = CTR × CVR, entire-space training eliminates SSB |
| **MMoE** | Shared experts with task-specific gating |
| **PLE** | Task-specific + shared experts, progressive extraction |

## Notebooks

1. **Data Exploration**: Parse Ali-CCP format, conversion funnel, SSB analysis
2. **ESMM**: Implement and train ESMM, compare with Naive CVR
3. **MMoE & PLE**: Advanced multi-task architectures
4. **Comparison**: Head-to-head evaluation of all models
