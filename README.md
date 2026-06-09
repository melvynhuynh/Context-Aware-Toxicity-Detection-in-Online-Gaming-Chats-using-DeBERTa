# Context-Aware Toxicity Detection in Online Gaming Chats using DeBERTa

**EE-559 Deep Learning — Group Mini-Project (Group 27)**
Melvyn Huynh · Anna Lekontseva · Yann Martin Thomé

This repository contains the training and analysis code for our study of **how context length and speaker identity affect per-class toxicity detection** in DOTA 2 / League-style in-game chat. We fine-tune `DeBERTa-v3-base` on the [CONDA](https://aclanthology.org/2021.acl-long.132/) benchmark and run a two-axis ablation over context-window size and speaker-policy filter.

- 🤗 **Models:** https://huggingface.co/Yanno72
- 🚀 **Live demo (Spaces):** https://huggingface.co/spaces/Yanno72/conda-ksweep

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| **Code** — training notebooks + analysis | [`context_length_sweep/`](context_length_sweep/), [`speaker_policy_ablation/`](speaker_policy_ablation/), [`figures/`](figures/) |
| **Demo app** — interactive Streamlit k-sweep | [`Website/`](Website/) · [hosted on Spaces](https://huggingface.co/spaces/Yanno72/conda-ksweep) |
| **Screencast** — demo running locally | [`video_of_code_running/Video_website_running_local.mov`](video_of_code_running/) |
| **Poster** | `rendu/EE559Poster_group27.pdf` *(submitted separately)* |
| **Paper** — 3-page IEEE report | *(submitted separately)* |
| **Trained models** | [HuggingFace Hub — Yanno72](https://huggingface.co/Yanno72) |

---

## Problem & objective

In-game chat moderation is a deployment-relevant NLP problem: toxic intent often surfaces as *implicit* sarcasm or kill-shaming (`"ez mid"`, `"feed"`) whose toxic reading depends on the conversational context. We identify three gaps in prior context-aware work and address each:

1. **Per-class blindness** — prior studies aggregate over binary / type-based labels, hiding which class drives the contextual gain. → We run a **per-class** ablation on CONDA's four intent labels.
2. **Context-source puzzle** — which source of context (own history, teammates, opponents) carries the signal is uncharacterized. → We add a **speaker-policy decomposition**.
3. **Mechanism opacity** — F1 deltas are reported without explaining *why*. → We analyze what property of preceding messages produces the gain.

**Central hypothesis (tested and refuted):** that context preferentially resolves the ambiguous *Implicit* class. We find the gain is instead **class-uniform** — context acts as a generic disambiguation signal.

## Task formulation

4-way intent classification over CONDA labels: **E** (Explicit), **I** (Implicit), **A** (Action), **O** (Other).

Given a target utterance `u_t` from player `P0` and a context window `C_t^k` of the `k` most recent prior messages, the input sequence is:

```
[CLS]  π_s( C_t^k )  [SEP]  "P0: u_t"  [SEP]
```

where `π_s` is the speaker-policy filter. Each message is prefixed with its `P{slot}:` tag and the sequence is left-truncated to preserve the target utterance. Speaker identity is encoded **lexically** via the `P{slot}:` tags.

## Two-axis ablation

| Axis | Values |
|------|--------|
| **Context length** `k` | `0, 1, 3, 5, 10` |
| **Speaker policy** `s` | `mixed` (all prior), `same` (P0 only), `team` (P0's team excl. P0), `enemy` (opposing team), `other` (all excl. P0 = team ∪ enemy) |

At `k=0` the filter is a no-op, so the five policies share one baseline → **1 + 5×4 = 21 conditions**.

> Note: in the notebooks the speaker policies are named `same_speaker`, `other_speaker`, `ally_team` (= `team`), and `enemy_team` (= `enemy`). DOTA team is derived as `playerSlot // 5` (slots 0–4 = Radiant, 5–9 = Dire).

## Model & training

- **Backbone:** `microsoft/deberta-v3-base` (~184M params, disentangled attention), linear classification head over `[CLS]` → 4-way softmax.
- **Loss:** cross-entropy, **no class re-weighting** (matches CONDA's protocol; class distribution is skewed 74/13/6/6 % O/E/I/A).
- **Optimizer:** AdamW, lr `2e-5`, weight decay `0.01`, 5 epochs, batch size 16, 10% warmup, early stopping on macro-F1.
- **Seeds:** the mixed-policy context-length sweep uses **three seeds** `{7, 27, 42}` for variance; the speaker-policy ablations (`k ≥ 1`) use a **single seed (42)** due to compute, so their differences are reported as *indicative*.

## Repository structure

```
.
├── context_length_sweep/        # Mixed-policy context-length sweep (3 seeds)
│   ├── conda_k0.ipynb           #   no-context baseline
│   ├── conda_k1.ipynb
│   ├── conda_k3.ipynb
│   ├── conda_k5.ipynb
│   └── conda_k10.ipynb
├── speaker_policy_ablation/     # Speaker-policy decomposition (single seed 42)
│   ├── conda_k{1,3,5,10}_same_speaker*.ipynb
│   ├── conda_k{1,3,5,10}_other_speaker*.ipynb
│   ├── conda_k{1,3,5,10}_ally_team*.ipynb   # = "team" policy
│   └── conda_k{1,3,5,10}_enemy_team*.ipynb  # = "enemy" policy
├── figures/                     # Plots & analysis for the paper/poster
│   ├── figures_report.ipynb
│   └── piste1_slot_density.ipynb
├── Website/                     # Streamlit demo (the screencast deliverable)
│   ├── app.py
│   ├── requirements.txt
│   ├── meta_cache/              # cached eval metrics per (seed, k) for the plots
│   └── README.md
└── video_of_code_running/       # Screencast of the demo running locally
    └── Video_website_running_local.mov
```

Each training notebook is self-contained and follows the same flow: load the CONDA split → build the (filtered) context index → tokenize with `P{slot}:` tags and `[SEP]` separators → fine-tune `DeBERTa-v3-base` → report macro- and per-class F1 → export the model + `meta.json`. The notebooks were authored for **Google Colab (T4 GPU)**; the `k` and `KIND` constants at the top select the condition.

## Interactive demo

[`Website/`](Website/) is a **Streamlit** app that loads the five fine-tuned models (`k = 0, 1, 3, 5, 10`) side-by-side: type a chat turn with optional context and watch how each `k` classifies it, with per-class probability bars, the multi-seed F1-vs-`k` plot, the speaker-policy ablation table, and a validation-metrics heatmap. A screencast of it running locally is in [`video_of_code_running/`](video_of_code_running/).

- **Hosted (no install):** https://huggingface.co/spaces/Yanno72/conda-ksweep
- **Local:**
  ```bash
  cd Website
  python3.11 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/streamlit run app.py
  ```
  Models are pulled from the HuggingFace Hub on first inference (~2.5 min total); use **🔥 Pre-load all models** in the sidebar to warm them. See [`Website/README.md`](Website/README.md) for details.

> ⚠️ **Reproducibility:** `requirements.txt` pins **`transformers==5.1.0`**. Newer 5.8.x versions have a DeBERTa-v2 CPU forward-pass regression that silently drops macro-F1 from ~0.83 to ~0.21 on the same weights — do not upgrade without re-checking the numbers.

## Data

[CONDA](https://github.com/usydnlp/CONDA) (Weld et al., 2021): 44,869 utterances of in-game DOTA 2 chat, dual-annotated with four intent classes. We use the official train/valid split (**26,921 / 8,974** utterances) and report all metrics on validation.

The notebooks expect `CONDA_train.csv` and `CONDA_valid.csv` (uploaded via the Colab file picker) with columns `conversationId`, `chatTime`, `playerSlot`, `utterance`, and `intentClass`. Rows are sorted by `(conversationId, chatTime)` so the context lookup is correct.

## How to reproduce

1. Open a notebook in Google Colab and set the runtime to **GPU (T4)**.
2. Run the install cell (`sentencepiece`, `protobuf` — required for DeBERTa-v3's tokenizer).
3. Upload `CONDA_train.csv` / `CONDA_valid.csv` when prompted.
4. Set `K` (and `KIND` for the speaker-policy notebooks), then run all cells.
5. The trained model and `meta.json` are saved to Colab local disk and archived to `k{K}_model.tar.gz` for download.

To reproduce the full grid, run each `k` in `context_length_sweep/` (×3 seeds for variance) and each `(k, policy)` combination in `speaker_policy_ablation/`.

## Key results

**Macro-F1 by context length `k` and speaker policy `s`** (mixed = mean ± std over seeds {7, 27, 42}; policy-specific entries are single-seed 42):

| `k` | mixed | same | team | enemy | other |
|----|-------|------|------|-------|-------|
| 0  | 0.825 ± 0.004 | — | — | — | — |
| 1  | 0.837 ± 0.003 | 0.837 | 0.837 | 0.836 | 0.837 |
| 3  | 0.839 ± 0.002 | 0.837 | 0.833 | 0.840 | 0.840 |
| 5  | 0.825 ± 0.007 | 0.838 | 0.835 | 0.837 | 0.823 |
| 10 | 0.828 ± 0.013 | 0.831 | 0.838 | 0.834 | 0.835 |

1. **Context gives a modest, class-uniform gain at `k=1`.** F1(Implicit) rises +0.020 (0.740 → 0.760) and F1(Explicit) +0.019 (0.830 → 0.849) — essentially identical gains. This **refutes** the hypothesis that context preferentially helps the Implicit class; context acts as a generic disambiguation signal.
2. **Saturation is immediate.** `k=3` (0.839) is within seed noise of `k=1`; `k=5` regresses to the `k=0` baseline; `k=10` recovers only partially but with 5× the variance. Longer windows add noise, not signal — **`k ∈ {1, 3}` is Pareto-optimal**.
3. **Speaker policy steers precision/recall at constant F1.** All five policies cluster within a 0.835–0.840 macro-F1 band but occupy distinct operating points: *enemy-only* context maximizes Implicit precision (0.938), *other-speaker* maximizes Implicit recall (0.672) — a 9-point precision spread invisible to aggregate metrics. Speaker policy thus works as a **free moderation-threshold knob**. *(Single-seed; needs multi-seed confirmation.)*

## Limitations

- Three seeds establish non-overlapping ranges (`k=0` vs `k≥1`) but not tight intervals on the `k=1` vs `k=3` gap.
- Single architecture (`DeBERTa-v3-base`); scaling and other transformer families untested.
- Narrow domain (DOTA 2 chat only); not validated on broader hate-speech corpora.
- Speaker-policy ablations use a single seed, so their operating-point differences are indicative.

## Future work

Much DOTA 2 toxicity is triggered by **unobservable game events** (kills, item drops, hero picks) that chat context alone cannot recover, regardless of window size or speaker policy. Incorporating gameplay metadata is the most promising direction.

## References

1. Pavlopoulos et al., "Toxicity detection: Does context really matter?", *ACL* 2020.
2. Davidson et al., "Automated hate speech detection and the problem of offensive language," *ICWSM* 2017.
3. Weld et al., "CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection," *ACL-IJCNLP* 2021.
4. He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," *ICLR* 2023.
5. Zampieri et al., "Predicting the type and target of offensive posts in social media," *NAACL-HLT* 2019.
6. Gao and Huang, "Detecting online hate speech using context aware models," *RANLP* 2017.
7. Yang et al., "ToxBuster: In-game chat toxicity buster with an operator's perspective," Ubisoft La Forge, 2023.
