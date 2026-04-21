---
title: Dsan 6500 Project
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# Song Popularity Predictor

Predicts the Deezer chart popularity tier of a song using an Audio Spectrogram Transformer fused with track metadata.

## Model

**Architecture:** AST Fusion (4.9M params)
- Audio branch: 128×128 mel spectrogram → 16×16 patches → 6-layer Transformer encoder → 256-dim CLS embedding
- Metadata branch: genre (27-class one-hot) + gain, duration, contributors, track position, explicit → MLP → 64-dim embedding
- Late fusion → CORN ordinal head → 5 popularity buckets

**Performance on held-out test set:**
- Exact accuracy: 43.9%
- Within-1-bucket accuracy: 83.9%
- Ordinal MAE: 0.75

**Training data:** 7,549 Deezer chart tracks across 27 genres

## Popularity Buckets

| Bucket | Tiers | Deezer Rank Range |
|--------|-------|-------------------|
| Very High Popularity | 1–2 | 0 – 200k |
| High Popularity | 3–4 | 200k – 400k |
| Moderate Popularity | 5–6 | 400k – 600k |
| Low Chart Presence | 7–8 | 600k – 800k |
| Very Low Chart Presence | 9–10 | 800k – 1M |

## Deploying to HF Spaces

1. Generate the scaler from the project root: `python demo/save_scaler.py`
2. Copy `transformer_model/best_ast_model.pt` and `demo/scaler.pkl` into this directory
3. Create a new HF Space (SDK: Gradio) and push this directory
