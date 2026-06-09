---
title: CONDA k-Sweep
emoji: 🎮
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
short_description: Context-window ablation for in-game toxicity classification
---

# CONDA k-Sweep — context-window ablation

Streamlit demo accompanying our EE-559 group project *Context-Aware Toxicity
Detection in Online Gaming Chats using DeBERTa* (Group 27).

Five DeBERTa-v3-base classifiers fine-tuned on the CONDA DOTA 2 / League of
Legends chat dataset with **k = 0, 1, 3, 5, 10** prior messages of context.
Type a chat turn (with optional context) and see how each k classifies it
side-by-side.

**Headline finding:** the k = 0 → k = 1 gain is class-uniform (+0.020 F1 on
Implicit, +0.019 on Explicit), saturates at k = 1, and grows seed-sensitive
at k ≥ 5. Speaker policy steers the precision–recall trade-off at constant
macro-F1.

---

## Two ways to try it

### Option 1 — Hosted version (no install needed)

Just open: **https://huggingface.co/spaces/Yanno72/conda-ksweep**

The first visitor after a period of inactivity may wait ~1–2 minutes for
the Space to wake up and cache the models. Subsequent visits are instant.

### Option 2 — Run it locally

Requires Python 3.11+ and ~3 GB of free disk (for the model cache on first
launch).

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Launch
.venv/bin/streamlit run app.py
```

Streamlit will print `Local URL: http://localhost:8501` — open that in
your browser.

**First-launch latency.** The 5 fine-tuned models are downloaded from
HuggingFace Hub on first inference (~30 s per model, ~2.5 minutes total).
Click **🔥 Pre-load all models** in the sidebar to warm them up. Subsequent
runs use the local HuggingFace cache.

---

## What you'll see

- **F1 vs context-window plot** with multi-seed error bars (mean ± min–max
  across 3 seeds), including a dashed macro-F1 line
- **4 preset examples** (one per class: I, E, O, A) — click "Classify" to
  run inference across all 5 models simultaneously
- **Per-model predictions** with class probability bars and the exact input
  the model sees
- **Speaker-policy ablation table** showing how the 5 speaker policies
  cluster within a 0.835–0.840 macro-F1 band but produce distinct
  precision–recall operating points
- **Multi-seed validation metrics** as a heatmap (mean ± std per cell)

---

## Models hosted at

- [Yanno72/conda-k0](https://huggingface.co/Yanno72/conda-k0)
- [Yanno72/conda-k1](https://huggingface.co/Yanno72/conda-k1)
- [Yanno72/conda-k3](https://huggingface.co/Yanno72/conda-k3)
- [Yanno72/conda-k5](https://huggingface.co/Yanno72/conda-k5)
- [Yanno72/conda-k10](https://huggingface.co/Yanno72/conda-k10)

---

## Reproducibility note

`requirements.txt` pins **`transformers==5.1.0`**. Newer versions (5.8.x)
introduce a DeBERTa-v2 CPU forward-pass regression that silently produces
incorrect predictions on the same model weights (we measured macro F1
dropping from 0.83 to 0.21 under 5.8.1). Do not upgrade `transformers`
without re-verifying numbers against the validation set.

---

## Authors

Group 27 — Melvyn Huynh, Anna Lekontseva, Yann Martin Thomé
EE-559: Deep Learning, EPFL, 2026
