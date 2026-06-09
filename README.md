# Context-Aware Toxicity Detection in Online Gaming Chats using DeBERTa

**EE-559 Deep Learning, Group Mini-Project, Group 27**
Melvyn Huynh, Anna Lekontseva, Yann Martin Thomé

We fine-tune `DeBERTa-v3-base` on the **CONDA** in-game-chat benchmark and study how context length and speaker identity affect *per-class* toxicity detection in DOTA 2 / League-style chat. We run a two-axis ablation, context-window size `k` in `{0,1,3,5,10}` crossed with five speaker policies, and report per-class F1 across the resulting 21 conditions.

> **Headline finding:** context gives a modest, class-uniform gain at `k=1` (+0.020 F1 Implicit, +0.019 Explicit), saturates immediately, and regresses for `k >= 5`. Speaker policy shifts the precision/recall operating point at constant macro-F1.

## External resources (not stored in this repo)

To keep the repository light, **model weights and the dataset are not committed**. They are linked here instead:

| Resource | Where | Note |
|----------|-------|------|
| **Trained models** (k = 0,1,3,5,10) | [huggingface.co/Yanno72](https://huggingface.co/Yanno72): [k0](https://huggingface.co/Yanno72/conda-k0), [k1](https://huggingface.co/Yanno72/conda-k1), [k3](https://huggingface.co/Yanno72/conda-k3), [k5](https://huggingface.co/Yanno72/conda-k5), [k10](https://huggingface.co/Yanno72/conda-k10) | downloaded on demand by the demo / notebooks |
| **Dataset (CONDA)** | [github.com/usydnlp/CONDA](https://github.com/usydnlp/CONDA) (Weld et al., ACL-IJCNLP 2021) | not redistributed here; see [Dataset](#dataset) |
| **Live demo (Spaces)** | [huggingface.co/spaces/Yanno72/conda-ksweep](https://huggingface.co/spaces/Yanno72/conda-ksweep) | hosted Streamlit app |
| **Screencast of the demo** | [`video_of_code_running/Video_website_running_local.mov`](video_of_code_running/) | code running locally |

## Repository structure

```
.
├── context_length_sweep/        # Axis 1: mixed-policy context-length sweep (3 seeds)
│   ├── conda_k0.ipynb           #   k = 0, no-context baseline
│   ├── conda_k1.ipynb           #   k = 1
│   ├── conda_k3.ipynb           #   k = 3
│   ├── conda_k5.ipynb           #   k = 5
│   └── conda_k10.ipynb          #   k = 10
│
├── speaker_policy_ablation/     # Axis 2: speaker-policy decomposition (single seed 42)
│   ├── conda_k{1,3,5,10}_same_speaker*.ipynb    # "same":  P0's own prior messages
│   ├── conda_k{1,3,5,10}_other_speaker*.ipynb   # "other": everyone except P0
│   ├── conda_k{1,3,5,10}_ally_team*.ipynb       # "team":  P0's team, excl. P0
│   └── conda_k{1,3,5,10}_enemy_team*.ipynb      # "enemy": opposing team only
│
├── figures/                     # Plots and analysis for the paper / poster
│   ├── figures_report.ipynb     #   per-class F1 vs k, results tables
│   └── piste1_slot_density.ipynb#   slot-annotation / mechanism analysis
│
├── Website/                     # Interactive demo (Streamlit), see "Demo / Website"
│   ├── app.py                   #   the app
│   ├── requirements.txt         #   pinned deps (transformers==5.1.0, see warning)
│   ├── meta_cache/              #   cached eval metrics per (seed, k) for the plots
│   └── README.md                #   demo-specific instructions
│
├── video_of_code_running/       # Screencast deliverable
│   └── Video_website_running_local.mov
│
└── README.md                    # this file
```

Every training notebook is **self-contained** and follows the same flow: load the CONDA split, build the (speaker-filtered) context index, tokenize with `P{slot}:` tags and `[SEP]` separators, fine-tune `DeBERTa-v3-base`, report macro- and per-class F1, then export the model plus a `meta.json`. The condition is selected by the `K` (and `KIND`) constants at the top of each notebook. Notebooks were authored for **Google Colab (T4 GPU)**.

## Dataset

[**CONDA**](https://github.com/usydnlp/CONDA) (Weld et al., 2021): 44,869 utterances of in-game DOTA 2 chat, dual-annotated into **4 intent classes**: `E` Explicit, `I` Implicit, `A` Action, `O` Other (distribution 13 / 6 / 6 / 74 %). We use the official **train / valid split (26,921 / 8,974)** and report all metrics on validation; `k=0` (no context) is the baseline.

The dataset is **not redistributed here**. Download it from the [CONDA repository](https://github.com/usydnlp/CONDA) and provide two CSVs, `CONDA_train.csv` and `CONDA_valid.csv`, with columns `conversationId`, `chatTime`, `playerSlot`, `utterance`, `intentClass`. The notebooks sort rows by `(conversationId, chatTime)` so the context lookup is correct, and derive DOTA team as `playerSlot // 5` (slots 0 to 4 = Radiant, 5 to 9 = Dire).

## Method (summary)

**Input.** Given a target utterance `u_t` from player `P0` and the `k` most recent prior messages `C_t^k`, the model sees:

```
[CLS]  pi_s( C_t^k )  [SEP]  "P0: u_t"  [SEP]
```

where `pi_s` is the speaker-policy filter; each message is prefixed with its `P{slot}:` tag (speaker identity encoded lexically), left-truncated to preserve the target.

**Two-axis ablation**, giving `1 + 5x4 = 21` conditions:

| Axis | Values |
|------|--------|
| Context length `k` | `0, 1, 3, 5, 10` |
| Speaker policy `s` | `mixed` (all prior), `same` (P0 only), `team` (P0's team excl. P0), `enemy` (opposing team), `other` (all excl. P0, equal to team union enemy) |

> In the notebooks the policies are named `same_speaker`, `other_speaker`, `ally_team` (= `team`), `enemy_team` (= `enemy`). At `k=0` the filter is a no-op, so the five policies share one baseline.

**Model and training.** `microsoft/deberta-v3-base` (about 184M params) plus a linear head over `[CLS]`, 4-way softmax. Cross-entropy, **no class re-weighting** (matches CONDA's protocol). AdamW, lr `2e-5`, weight decay `0.01`, 5 epochs, batch 16, 10 % warmup, early stopping on macro-F1. The mixed-policy sweep uses **3 seeds `{7, 27, 42}`** for variance; speaker-policy ablations use a **single seed (42)** due to compute, so their differences are indicative.

## Demo / Website

[`Website/`](Website/) is a **Streamlit** app that loads the five fine-tuned models (`k = 0,1,3,5,10`) and runs them side-by-side: type a chat turn (with optional context) and see how each `k` classifies it, with per-class probability bars, the exact input each model sees, the multi-seed F1-vs-`k` plot, the speaker-policy ablation table, and a validation-metrics heatmap. The models are pulled from the HuggingFace Hub on first inference, so no weights are stored locally.

A screencast of it running is in [`video_of_code_running/Video_website_running_local.mov`](video_of_code_running/).

**Option 1, hosted (nothing to install):** open [huggingface.co/spaces/Yanno72/conda-ksweep](https://huggingface.co/spaces/Yanno72/conda-ksweep). The first visit after inactivity may take 1 to 2 min to wake the Space.

**Option 2, run locally** (Python 3.11+, about 3 GB free for the model cache):

```bash
cd Website
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py          # opens http://localhost:8501
```

On first inference the 5 models download from the Hub (about 2.5 min total); click **Pre-load all models** in the sidebar to warm them. See [`Website/README.md`](Website/README.md) for more.

> **Reproducibility warning:** `requirements.txt` pins **`transformers==5.1.0`**. Newer 5.8.x versions have a DeBERTa-v2 CPU forward-pass regression that silently drops macro-F1 from about 0.83 to about 0.21 on the same weights. Do **not** upgrade without re-verifying the numbers.

## Reproducing the experiments

1. Open a notebook in **Google Colab** and set the runtime to **GPU (T4)**.
2. Run the install cell (`sentencepiece`, `protobuf`, needed for DeBERTa-v3's tokenizer).
3. Upload `CONDA_train.csv` / `CONDA_valid.csv` when prompted.
4. Set `K` (and `KIND` for the speaker-policy notebooks), then **Run all**.
5. The best model plus `meta.json` are saved and archived to `k{K}_model.tar.gz` for download.

Full grid = each `k` in `context_length_sweep/` (times 3 seeds for variance) plus each `(k, policy)` in `speaker_policy_ablation/`.

## Results

**Macro-F1 by context length `k` and speaker policy `s`.** `mixed` is mean +/- std over seeds {7, 27, 42}; policy-specific entries are single-seed (42):

| `k` | mixed | same | team | enemy | other |
|----|-------|------|------|-------|-------|
| 0  | 0.825 +/- 0.004 | n/a | n/a | n/a | n/a |
| 1  | 0.837 +/- 0.003 | 0.837 | 0.837 | 0.836 | 0.837 |
| 3  | 0.839 +/- 0.002 | 0.837 | 0.833 | 0.840 | 0.840 |
| 5  | 0.825 +/- 0.007 | 0.838 | 0.835 | 0.837 | 0.823 |
| 10 | 0.828 +/- 0.013 | 0.831 | 0.838 | 0.834 | 0.835 |

1. **Class-uniform gain at `k=1`.** F1(Implicit) +0.020 and F1(Explicit) +0.019, essentially identical, which refutes the hypothesis that context preferentially resolves the Implicit class. Context acts as a generic disambiguation signal.
2. **Immediate saturation.** `k=3` is within seed noise of `k=1`; `k=5` regresses to baseline; `k=10` recovers only partially, with 5x the variance. `k` in `{1,3}` is Pareto-optimal.
3. **Speaker policy as a free operating-point selector.** All five policies sit in a 0.835 to 0.840 macro-F1 band but differ in precision/recall: `enemy` context maximizes Implicit precision (0.938), `other` maximizes Implicit recall (0.672), a 9-point spread invisible to aggregate metrics. (Single-seed; needs multi-seed confirmation.)

## Limitations

- 3 seeds give non-overlapping ranges (`k=0` vs `k>=1`) but not tight CIs on the `k=1` vs `k=3` gap.
- Single architecture (`DeBERTa-v3-base`); scaling and other families untested.
- Narrow domain (DOTA 2 chat); not validated on broader hate-speech corpora.
- Speaker-policy ablations are single-seed, so operating-point differences are indicative.

**Open question and future work.** Much DOTA 2 toxicity is triggered by unobservable game events (kills, item drops, hero picks) that chat context alone cannot recover. Incorporating gameplay metadata is the most promising direction.

## References

1. Pavlopoulos et al., "Toxicity detection: Does context really matter?", *ACL* 2020.
2. Davidson et al., "Automated hate speech detection and the problem of offensive language," *ICWSM* 2017.
3. Weld et al., "CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection," *ACL-IJCNLP* 2021.
4. He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," *ICLR* 2023.
5. Zampieri et al., "Predicting the type and target of offensive posts in social media," *NAACL-HLT* 2019.
6. Gao and Huang, "Detecting online hate speech using context aware models," *RANLP* 2017.
7. Yang et al., "ToxBuster: In-game chat toxicity buster with an operator's perspective," Ubisoft La Forge, 2023.

---

*Group 27, Melvyn Huynh, Anna Lekontseva, Yann Martin Thomé. EE-559 Deep Learning, EPFL, 2026.*
