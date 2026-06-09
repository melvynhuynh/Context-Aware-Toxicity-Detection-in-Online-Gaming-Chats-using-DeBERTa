"""
CONDA Toxicity Detection — Context Window k-Sweep Demo
======================================================
Streamlit research demo comparing five DeBERTa-v3-base models fine-tuned with
k = 0, 1, 3, 5, 10 prior chat messages of context on the CONDA League-of-Legends
in-game chat dataset.

Run:
    .venv/bin/streamlit run app.py

INPUT FORMAT (verified against conda_k*.ipynb cell 12 CONDAContextDataset)
- k = 0:               raw target text, no speaker tag.
- k > 0 + context:     "P3: prev1 [SEP] P7: prev2 [SEP] ... [SEP] P0: target"
- k > 0 + no context:  "P0: target"   (matches training for early msgs).
Do not change format_input() without re-verifying against the notebooks.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("./models")
# Additional per-seed folders, in priority order. The first folder is the one
# the live demo uses for inference; the others contribute only F1 numbers to
# the headline plot's error bars (range across seeds).
EXTRA_SEED_DIRS = [
    ("seed=42", Path("./models")),
    ("seed=7",  Path("./models_seed7")),
    ("seed=27", Path("./models_seed27")),
]
# HuggingFace Hub repo IDs — used as a fallback when local model dirs aren't
# present (i.e., when this app is running on Spaces / Streamlit Cloud).
HF_REPO = {
    0:  "Yanno72/conda-k0",
    1:  "Yanno72/conda-k1",
    3:  "Yanno72/conda-k3",
    5:  "Yanno72/conda-k5",
    10: "Yanno72/conda-k10",
}
# Small JSON cache holding meta.json per (seed, k) for the multi-seed headline
# plot when local seed folders aren't on disk.
META_CACHE_DIR = Path("./meta_cache")
K_VALUES = [0, 1, 3, 5, 10]
LABELS = ["E", "I", "A", "O"]
LABEL_NAMES = {
    "E": "Explicit toxicity",
    "I": "Implicit toxicity",
    "A": "Action / command",
    "O": "Other (non-toxic)",
}
LABEL_COLORS = {
    "E": "#b91c1c",  # red-700  — 6.0:1 on white (was #dc2626, 4.8:1)
    "I": "#b45309",  # amber-700 — 5.0:1 on white (was #d97706, 3.4:1, failed AA)
    "A": "#15803d",  # green-700 — 4.8:1 on white (was #16a34a, 3.0:1, failed AA)
    "O": "#475569",  # slate-600 — 7.4:1 on white (was #64748b, 4.6:1)
}
SEP = "[SEP]"

# Design tokens — Data-Dense Dashboard palette
COLOR_PRIMARY    = "#1e40af"
COLOR_SECONDARY  = "#3b82f6"
COLOR_ACCENT     = "#d97706"
COLOR_BG         = "#f8fafc"
COLOR_FG         = "#1e3a8a"
COLOR_MUTED      = "#e9eef6"
COLOR_BORDER     = "#dbeafe"

PRESETS: Dict[str, dict] = {
    "Implicit (I)": {
        "context": [(5, "0 10 score lol"), (3, "throw mid")],
        "target": "ez mid",
        "note": "Sarcastic kill-shaming: target speaker mocks a mid-laner doing 0/10. "
                "k>0 should pick up Implicit (I) once the failure context is visible.",
    },
    "Explicit (E)": {
        "context": [(5, "feeding all game"), (3, "report support")],
        "target": "you are trash kys uninstall",
        "note": "Direct harassment with slur and \"kys\" (kill yourself). "
                "Every k should predict Explicit (E) — slur is lexically obvious.",
    },
    "Other (O)": {
        "context": [(5, "gg wp"), (3, "good game everyone")],
        "target": "anyone else watching the major tonight",
        "note": "Off-topic post-match banter. Target should remain Other (O) "
                "regardless of context window.",
    },
    "Action (A)": {
        "context": [(5, "they smurfing fr"), (3, "5 stack pro players")],
        "target": "report those guys",
        "note": "Imperative game command — players asking to report suspected smurfs. "
                "Should predict Action (A) on k>0.",
    },
}

HEADLINE_FALLBACK = pd.DataFrame({
    "k":        [0,     1,     3,     5,     10],
    "E":        [0.819, 0.850, 0.846, 0.834, 0.844],
    "I":        [0.735, 0.762, 0.755, 0.736, 0.746],
    "A":        [0.778, 0.797, 0.807, 0.785, 0.790],
    "O":        [0.948, 0.952, 0.951, 0.949, 0.950],
    "Macro F1": [0.820, 0.840, 0.840, 0.826, 0.833],
    "Accuracy": [0.908, 0.917, 0.916, 0.911, 0.915],
}).set_index("k")


# ─────────────────────────────────────────────────────────────────────────────
# Model discovery & loading
# ─────────────────────────────────────────────────────────────────────────────
def available_models() -> Dict[int, str]:
    """Returns {k: source} where source is either a local Path (dev) or a
    HuggingFace Hub repo ID (deployment). Local dirs take priority."""
    found: Dict[int, str] = {}
    for k in K_VALUES:
        p = MODELS_DIR / f"k{k}_model"
        if p.exists() and (p / "config.json").exists():
            found[k] = str(p)
        elif k in HF_REPO:
            found[k] = HF_REPO[k]   # fallback to HF Hub
    return found


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Accepts either a local path or an HF Hub repo ID.
    `from_pretrained` handles both transparently (Hub repos are cached
    locally on first load)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # meta.json: try local first; on HF Hub, fall back to the cached copy.
    meta = {}
    local = Path(model_path) / "meta.json"
    if local.exists():
        meta = json.loads(local.read_text())
    else:
        cache = META_CACHE_DIR / f"seed42_k{_k_from_repo(model_path)}.json"
        if cache.exists():
            meta = json.loads(cache.read_text())
    return tokenizer, model, meta


def _k_from_repo(repo_or_path: str) -> str:
    """Extract the k value from a repo ID like 'Yanno72/conda-k3' or path."""
    name = str(repo_or_path).rstrip("/").split("/")[-1]
    return name.replace("conda-k", "").replace("k", "").replace("_model", "")


def read_all_metas(models_found: Dict[int, str]) -> Dict[int, dict]:
    metas: Dict[int, dict] = {}
    for k, src in models_found.items():
        meta = {}
        local = Path(src) / "meta.json"
        if local.exists():
            try:
                meta = json.loads(local.read_text())
            except Exception:
                pass
        else:
            cache = META_CACHE_DIR / f"seed42_k{k}.json"
            if cache.exists():
                try:
                    meta = json.loads(cache.read_text())
                except Exception:
                    pass
        metas[k] = meta
    return metas


def read_multi_seed_metas() -> Dict[str, Dict[int, dict]]:
    """Returns {seed_label: {k: meta_dict}} for every seed folder that exists.
    Local seed folders take priority; falls back to bundled meta_cache/ JSON
    files when running deployed (no local model dirs)."""
    out: Dict[str, Dict[int, dict]] = {}
    for label, root in EXTRA_SEED_DIRS:
        per_k: Dict[int, dict] = {}
        # Try local model dirs first
        if root.exists():
            for k in K_VALUES:
                mp = root / f"k{k}_model" / "meta.json"
                if mp.exists():
                    try:
                        per_k[k] = json.loads(mp.read_text())
                    except Exception:
                        pass
        # Fall back to bundled meta_cache/ (deployment scenario)
        if not per_k and META_CACHE_DIR.exists():
            seed_num = label.split("=")[-1]
            for k in K_VALUES:
                cache = META_CACHE_DIR / f"seed{seed_num}_k{k}.json"
                if cache.exists():
                    try:
                        per_k[k] = json.loads(cache.read_text())
                    except Exception:
                        pass
        if per_k:
            out[label] = per_k
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def format_input(target: str, context_msgs: List[Tuple[int, str]], k: int) -> str:
    if k == 0:
        return target
    if not context_msgs:
        return f"P0: {target}"
    used = context_msgs[-k:]
    parts = [f"P{slot}: {msg}" for slot, msg in used]
    parts.append(f"P0: {target}")
    return f" {SEP} ".join(parts)


def predict(model, tokenizer, text: str, max_length: int = 256):
    raw = tokenizer(text, add_special_tokens=True)["input_ids"]
    n_tokens = len(raw)
    truncated = n_tokens > max_length
    enc = tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length,
    )
    enc = {kk: v.to(model.device) for kk, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    id2label = model.config.id2label
    probs_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return probs_dict, n_tokens, truncated


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CONDA · k-Sweep",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pro CSS — Fira Code + Fira Sans + Data-Dense Dashboard tokens.
# Hides default Streamlit chrome and applies the design system globally.
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    :root {{
      --c-primary:   {COLOR_PRIMARY};
      --c-secondary: {COLOR_SECONDARY};
      --c-accent:    {COLOR_ACCENT};
      --c-bg:        {COLOR_BG};
      --c-fg:        {COLOR_FG};
      --c-muted:     {COLOR_MUTED};
      --c-border:    {COLOR_BORDER};
      --c-surface:   #ffffff;
      --c-text:      #0f172a;
      --c-text-2:    #475569;
      --c-success:   #16a34a;
      --c-warn:      #d97706;
      --c-danger:    #dc2626;
    }}

    #MainMenu, footer, header[data-testid='stHeader'] {{ visibility: hidden; }}
    .stApp {{ background: var(--c-bg); }}

    html, body, [class*="css"], div, p, label {{
      font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
      color: var(--c-text);
    }}
    /* Plain spans inherit Fira Sans from parents — but icon-font spans must keep
       their own font (Material Symbols), otherwise Streamlit's chevrons, ×, ▼,
       arrows render as raw text codes like 'keyboard_arrow_right'.
       Streamlit emotion-hashes the icon span's class so we can't rely on a
       class match — also target by position (first child of summary) and by
       aria-hidden (icons are decorative). */
    [class*="material-symbols"],
    span[class*="MaterialSymbols"],
    span[data-testid*="stIcon"],
    span[data-testid*="Icon"],
    [data-testid="stExpander"] summary > span:first-child,
    [data-testid="stExpander"] summary span[aria-hidden="true"],
    summary span[aria-hidden="true"],
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined {{
      font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                   'Material Icons', sans-serif !important;
      font-feature-settings: 'liga' 1 !important;
    }}

    h1, h2, h3, h4, h5 {{
      font-family: 'Fira Code', 'IBM Plex Mono', monospace !important;
      font-weight: 600 !important;
      letter-spacing: -0.01em;
      color: var(--c-fg) !important;
    }}
    h1 {{ font-size: 1.75rem !important; line-height: 1.2; }}
    h2 {{ font-size: 1.15rem !important; }}
    h3 {{
      font-size: 0.75rem !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--c-text-2) !important;
      margin-bottom: 0.5rem !important;
    }}

    .block-container {{
      max-width: 1400px;
      padding-top: 1.5rem !important;
      padding-bottom: 4rem !important;
    }}

    /* Buttons */
    .stButton > button {{
      font-family: 'Fira Sans', sans-serif !important;
      font-weight: 500 !important;
      font-size: 0.85rem !important;
      border-radius: 6px !important;
      border: 1px solid var(--c-border) !important;
      background: var(--c-surface) !important;
      color: var(--c-text) !important;
      transition: all 150ms ease-out !important;
      padding: 0.45rem 0.85rem !important;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .stButton > button:hover {{
      border-color: var(--c-primary) !important;
      color: var(--c-primary) !important;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(30, 64, 175, 0.10);
    }}
    .stButton > button:focus-visible {{
      outline: 2px solid var(--c-primary) !important;
      outline-offset: 2px;
    }}
    .stButton > button[kind="primary"] {{
      background: #bfdbfe !important;       /* blue-200 — soft pastel */
      border-color: #93c5fd !important;      /* blue-300 border */
      color: #1e3a8a !important;             /* blue-900 text, 7.8:1 contrast */
    }}
    .stButton > button[kind="primary"]:hover {{
      background: #93c5fd !important;        /* blue-300 on hover */
      border-color: #60a5fa !important;      /* blue-400 border on hover */
      color: #1e3a8a !important;
    }}

    .stDownloadButton > button {{
      font-family: 'Fira Sans', sans-serif !important;
      font-weight: 600 !important;
      font-size: 0.85rem !important;
      border-radius: 6px !important;
      border: 1px solid var(--c-border) !important;
      background: var(--c-surface) !important;
      color: var(--c-text) !important;
      padding: 0.5rem 1rem !important;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }}
    .stDownloadButton > button:hover {{
      border-color: var(--c-primary) !important;
      color: var(--c-primary) !important;
    }}

    /* Expander headers + body — force light surface + dark text. Streamlit's
       default uses theme-aware coloring that can land black-on-near-black
       depending on browser/OS dark-mode preference. We pin both. */
    [data-testid="stExpander"] {{
      background: var(--c-surface) !important;
    }}
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details > summary {{
      background: var(--c-surface) !important;
    }}
    /* Header text only — explicitly target the markdown container that holds
       the label, NOT the whole summary (which contains the chevron icon span).
       Setting font-family on summary would leak via inheritance onto an icon
       span that has an emotion-hashed class (no "material" in it). */
    [data-testid="stExpander"] summary {{
      color: #0f172a !important;
      background: transparent !important;
    }}
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] * {{
      color: #0f172a !important;
      font-weight: 600 !important;
      font-size: 0.9rem !important;
      font-family: 'Fira Sans', sans-serif !important;
      background: transparent !important;
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
      color: #0f172a !important;
    }}

    /* Code blocks inside expanders (and elsewhere): force light surface +
       dark text so 'P3: wow nice [SEP] P0: ez mid' is actually readable.
       Streamlit's default code theme is OS-theme-aware and can render black-on-black. */
    [data-testid="stCode"],
    [data-testid="stCodeBlock"],
    div[data-testid="stCode"],
    pre,
    pre[class*="language"],
    .stCodeBlock {{
      background: #f1f5f9 !important;
      border: 1px solid var(--c-border) !important;
      border-radius: 6px !important;
    }}
    [data-testid="stCode"] :not([class*="material"]):not([class*="MaterialSymbols"]):not([class*="icon"]),
    [data-testid="stCodeBlock"] :not([class*="material"]):not([class*="MaterialSymbols"]):not([class*="icon"]),
    pre :not([class*="material"]):not([class*="MaterialSymbols"]):not([class*="icon"]),
    code:not([class*="material"]):not([class*="MaterialSymbols"]):not([class*="icon"]) {{
      color: #0f172a !important;
      background: transparent !important;
      font-family: 'Fira Code', monospace !important;
    }}

    /* Streamlit help-icon tooltips: my global "color: var(--c-text)" rule was
       making dark-on-dark inside Streamlit's native tooltip popover. Force
       light text on the tooltip surface. */
    [data-baseweb="tooltip"],
    [data-baseweb="tooltip"] *,
    div[role="tooltip"],
    div[role="tooltip"] * {{
      color: #ffffff !important;
      font-family: 'Fira Sans', sans-serif !important;
      font-size: 0.8rem !important;
    }}

    /* Sidebar button label — bump weight so it stands out on the slate-50 bg */
    section[data-testid="stSidebar"] .stButton > button {{
      font-weight: 600 !important;
      color: #0f172a !important;
    }}

    /* Inputs */
    .stTextInput input, .stNumberInput input {{
      font-family: 'Fira Code', monospace !important;
      font-size: 0.85rem !important;
      border-radius: 6px !important;
      border: 1px solid var(--c-border) !important;
      background: var(--c-surface) !important;
      color: var(--c-text) !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
      border-color: var(--c-primary) !important;
      box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.12) !important;
      outline: none !important;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
      border: 1px solid var(--c-border) !important;
      border-radius: 8px;
      background: var(--c-surface) !important;
    }}
    [data-testid="stDataFrame"] *:not([class*="material"]):not([class*="MaterialSymbols"]):not([class*="icon"]) {{
      font-family: 'Fira Code', monospace !important;
      font-size: 0.8rem !important;
      color: #0f172a !important;
    }}
    /* Column headers — explicit weight + dark color so they don't fade to grey */
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="columnheader"] * {{
      color: #0f172a !important;
      font-weight: 600 !important;
      background: #f1f5f9 !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
      background: #f1f5f9 !important;
      border-right: 1px solid var(--c-border) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
      font-size: 0.85rem;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
      border: 1px solid var(--c-border) !important;
      border-radius: 8px !important;
      background: var(--c-surface);
    }}
    [data-testid="stExpander"] summary {{
      font-family: 'Fira Sans', sans-serif !important;
      font-weight: 500 !important;
      font-size: 0.85rem !important;
      color: var(--c-fg) !important;
    }}

    /* Custom components */
    .pred-card {{
      padding: 18px 20px;
      border-radius: 8px;
      border-left: 4px solid var(--c-primary);
      background: var(--c-surface);
      border-top: 1px solid var(--c-border);
      border-right: 1px solid var(--c-border);
      border-bottom: 1px solid var(--c-border);
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
      margin-bottom: 0.5rem;
    }}
    .pred-card-label {{
      font-family: 'Fira Code', monospace;
      font-size: 1.75rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 1;
    }}
    .pred-card-name {{
      font-family: 'Fira Sans', sans-serif;
      font-size: 0.85rem;
      color: var(--c-text-2);
      margin-top: 4px;
    }}
    .pred-card-conf {{
      font-family: 'Fira Code', monospace;
      font-size: 0.7rem;
      color: var(--c-text-2);
      margin-top: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    .kpi-card {{
      padding: 14px 16px;
      border-radius: 8px;
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      transition: all 150ms ease-out;
    }}
    .kpi-card:hover {{
      border-color: var(--c-primary);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(30, 64, 175, 0.06);
    }}
    .kpi-label {{
      font-family: 'Fira Sans', sans-serif;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--c-text-2);
      font-weight: 500;
    }}
    .kpi-value {{
      font-family: 'Fira Code', monospace;
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--c-fg);
      margin-top: 4px;
      line-height: 1.1;
    }}
    .kpi-sub {{
      font-family: 'Fira Code', monospace;
      font-size: 0.7rem;
      color: var(--c-text-2);
      margin-left: 0.25rem;
    }}

    .target-pill, .ctx-pill {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 4px;
      font-family: 'Fira Code', monospace;
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      vertical-align: middle;
    }}
    .target-pill {{
      background: #b45309;  /* amber-700 — 5.0:1 with white text */
      color: white;
    }}
    .ctx-pill {{
      background: var(--c-muted);
      color: var(--c-text-2);
      border: 1px solid var(--c-border);
    }}

    .small-mono {{
      font-family: 'Fira Code', monospace !important;
      font-size: 0.75rem;
      color: var(--c-text-2);
    }}

    .banner {{
      padding: 12px 16px;
      border-radius: 6px;
      font-family: 'Fira Sans', sans-serif;
      font-size: 0.85rem;
      line-height: 1.5;
      margin: 0.5rem 0;
    }}
    .banner-success {{
      background: rgba(22, 163, 74, 0.08);
      border-left: 3px solid var(--c-success);
      color: #14532d;
    }}
    .banner-warn {{
      background: rgba(217, 119, 6, 0.08);
      border-left: 3px solid var(--c-accent);
      color: #78350f;
    }}
    .banner-info {{
      background: rgba(59, 130, 246, 0.06);
      border-left: 3px solid var(--c-secondary);
      color: var(--c-fg);
    }}

    hr {{ border-color: var(--c-border) !important; margin: 1.25rem 0 !important; }}
    .section-spacer {{ height: 1.25rem; }}

    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: 0.001ms !important;
        transition-duration: 0.001ms !important;
      }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 1.25rem; border-bottom: 1px solid var(--c-border); padding-bottom: 1rem;">
      <div style="display: flex; align-items: baseline; gap: 0.75rem;">
        <h1 style="margin: 0;">CONDA · Context-Window k-Sweep</h1>
        <span class="small-mono" style="color: var(--c-text-2);">v0.1 · research demo</span>
      </div>
      <div style="color: var(--c-text-2); font-size: 0.95rem; margin-top: 0.35rem;">
        Five DeBERTa-v3-base classifiers fine-tuned on the CONDA League-of-Legends chat dataset with
        <span style="font-family: 'Fira Code', monospace; color: var(--c-fg); font-weight: 600; background: var(--c-muted); padding: 1px 6px; border-radius: 3px;">k = 0, 1, 3, 5, 10</span>
        prior messages of context.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

models_found = available_models()
all_metas = read_all_metas(models_found)


# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
def kpi_row():
    k0_macro = all_metas.get(0, {}).get("eval_metrics", {}).get("eval_f1_macro")
    best_k, best_macro = None, -1.0
    for k, m in all_metas.items():
        v = m.get("eval_metrics", {}).get("eval_f1_macro")
        if v is not None and v > best_macro:
            best_macro = v
            best_k = k
    best_f1_I = max(
        (m.get("eval_metrics", {}).get("eval_f1_I", 0) for m in all_metas.values()),
        default=0,
    )

    cols = st.columns(4)
    cols[0].markdown(
        f'<div class="kpi-card"><div class="kpi-label">Models loaded</div>'
        f'<div class="kpi-value">{len(models_found)}<span class="kpi-sub">/ 5</span></div></div>',
        unsafe_allow_html=True,
    )
    if best_macro >= 0:
        cols[1].markdown(
            f'<div class="kpi-card"><div class="kpi-label">Best macro F1</div>'
            f'<div class="kpi-value">{best_macro:.3f}<span class="kpi-sub">@ k={best_k}</span></div></div>',
            unsafe_allow_html=True,
        )
    if k0_macro is not None and best_macro >= 0:
        delta = best_macro - k0_macro
        color = "var(--c-success)" if delta >= 0 else "var(--c-danger)"
        sign = "+" if delta >= 0 else ""
        cols[2].markdown(
            f'<div class="kpi-card"><div class="kpi-label">Context gain Δ</div>'
            f'<div class="kpi-value" style="color: {color};">{sign}{delta:.3f}</div></div>',
            unsafe_allow_html=True,
        )
    if best_f1_I > 0:
        cols[3].markdown(
            f'<div class="kpi-card"><div class="kpi-label">Best F1(I) — implicit</div>'
            f'<div class="kpi-value" style="color: var(--c-accent);">{best_f1_I:.3f}</div></div>',
            unsafe_allow_html=True,
        )


kpi_row()
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Headline plot
# ─────────────────────────────────────────────────────────────────────────────
def build_headline_df():
    """Long-format df with columns: k, class, F1, seed.
    The 'class' column holds one of {E, I, A, O, macro}.
    Aggregated downstream into mean/min/max for the headline plot."""
    multi = read_multi_seed_metas()
    rows = []
    for seed_label, per_k in multi.items():
        for k, m in per_k.items():
            em = m.get("eval_metrics", {})
            for lbl in LABELS:
                v = em.get(f"eval_f1_{lbl}")
                if v is not None:
                    rows.append({"k": k, "class": lbl, "F1": float(v), "seed": seed_label})
            macro = em.get("eval_f1_macro")
            if macro is not None:
                rows.append({"k": k, "class": "macro", "F1": float(macro), "seed": seed_label})
    if rows:
        return pd.DataFrame(rows), "meta"
    # Fallback: PROJECT_CONTEXT.md numbers (single seed implicit).
    rows = []
    for k in HEADLINE_FALLBACK.index:
        for lbl in LABELS:
            rows.append({"k": int(k), "class": lbl, "F1": float(HEADLINE_FALLBACK.loc[k, lbl]),
                         "seed": "seed=42"})
        rows.append({"k": int(k), "class": "macro",
                     "F1": float(HEADLINE_FALLBACK.loc[k, "Macro F1"]),
                     "seed": "seed=42"})
    return pd.DataFrame(rows), "fallback"


head_df, source_kind = build_headline_df()
seed_count = head_df["seed"].nunique()

# Aggregate to mean/min/max per (k, class) — gives us error-bar arms.
agg = (head_df.groupby(["k", "class"])["F1"]
              .agg(["mean", "min", "max"])
              .reset_index())

st.markdown("### F1 per class vs context window k")

fig = go.Figure()
for cls in LABELS:
    sub = agg[agg["class"] == cls].sort_values("k")
    fig.add_trace(go.Scatter(
        x=sub["k"], y=sub["mean"],
        mode="lines+markers",
        name=cls,
        line=dict(color=LABEL_COLORS[cls], width=2.5),
        marker=dict(size=9, color=LABEL_COLORS[cls],
                    line=dict(width=2, color="white")),
        error_y=dict(
            type="data",
            symmetric=False,
            array=(sub["max"] - sub["mean"]).values,
            arrayminus=(sub["mean"] - sub["min"]).values,
            color=LABEL_COLORS[cls],
            thickness=1.5,
            width=4,
        ),
        hovertemplate=(
            f"<b>{cls}</b>  k=%{{x}}<br>"
            "mean F1 = %{y:.4f}<br>"
            "min–max across seeds<extra></extra>"
        ),
    ))

# Macro F1 — drawn last so it sits on top. Black, thicker, dashed to read
# as a different *kind* of metric (aggregate, not per-class).
macro_sub = agg[agg["class"] == "macro"].sort_values("k")
if not macro_sub.empty:
    fig.add_trace(go.Scatter(
        x=macro_sub["k"], y=macro_sub["mean"],
        mode="lines+markers",
        name="macro (mean of E/I/A/O)",
        line=dict(color="#0f172a", width=3.0, dash="dash"),
        marker=dict(size=10, color="#0f172a", symbol="diamond",
                    line=dict(width=2, color="white")),
        error_y=dict(
            type="data",
            symmetric=False,
            array=(macro_sub["max"] - macro_sub["mean"]).values,
            arrayminus=(macro_sub["mean"] - macro_sub["min"]).values,
            color="#0f172a",
            thickness=1.5,
            width=5,
        ),
        hovertemplate=(
            "<b>macro F1</b>  k=%{x}<br>"
            "mean = %{y:.4f}<br>"
            "min–max across seeds<extra></extra>"
        ),
    ))

fig.update_layout(
    title=None,
    xaxis=dict(
        tickmode="array", tickvals=[0, 1, 3, 5, 10],
        title=dict(text="<b>k</b>", font=dict(color="#0f172a")),
        gridcolor="#e2e8f0",
        showline=True, linecolor="#cbd5e1",
        zeroline=False,
        tickfont=dict(family="Fira Code, monospace", size=11, color="#0f172a"),
    ),
    yaxis=dict(
        range=[0.70, 0.97], tickformat=".2f",
        title=dict(text="<b>F1</b>", font=dict(color="#0f172a")),
        gridcolor="#e2e8f0",
        showline=True, linecolor="#cbd5e1",
        zeroline=False,
        tickfont=dict(family="Fira Code, monospace", size=11, color="#0f172a"),
    ),
    legend=dict(
        title_text="",
        orientation="h", x=0.5, xanchor="center", y=-0.22,
        font=dict(family="Fira Sans, sans-serif", size=12, color="#0f172a"),
    ),
    height=360,
    margin=dict(l=10, r=10, t=10, b=70),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Fira Sans, sans-serif", color="#0f172a"),
    hoverlabel=dict(font=dict(family="Fira Code, monospace", color="#0f172a"),
                    bgcolor="white", bordercolor="#cbd5e1"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if source_kind == "meta":
    seed_labels = sorted(head_df["seed"].unique())
    src_note = f"F1 from meta.json · {seed_count} seed{'s' if seed_count > 1 else ''} ({', '.join(seed_labels)})"
    if seed_count > 1:
        src_note += " · error bars show min–max range across seeds"
else:
    src_note = "From PROJECT_CONTEXT.md (meta.json unavailable)"
st.markdown(
    f'<div class="small-mono" style="text-align: right; margin-top: -0.5rem;">{src_note}</div>',
    unsafe_allow_html=True,
)

if seed_count >= 3:
    banner_msg = (
        f'<b>Reading the plot ({seed_count} seeds).</b> '
        'Context provides a modest, <b>class-uniform</b> gain: k=1 yields '
        '+0.020 F1(I) and +0.019 F1(E) — essentially identical absolute gains. '
        'Saturation is immediate (k=3 within noise of k=1); k≥5 degrades both '
        'classes uniformly. <b>k ∈ {1, 3} is Pareto-optimal</b>; the class-uniformity '
        'is unexpected — context acts as a generic disambiguation signal, not a '
        'preferential resolver of coded/implicit language.'
    )
elif seed_count > 1:
    banner_msg = (
        '<b>Reading the plot.</b> Adding <b>any</b> context (k=0 → k=1) gives a '
        'clean macro-F1 jump that is reproducible across seeds — the k=0 error '
        'bar lies entirely below the k=1 error bar (no overlap). k=1 through '
        'k=10 are all within seed noise — <b>no clear optimal k</b>. The '
        '"whether" of context matters more than the "how much".'
    )
else:
    banner_msg = (
        '<b>Reading the plot.</b> Adding <b>any</b> context (k=0 → k=1) gives a clean '
        '+1.5 to +2 macro F1 jump. k=1 through k=10 are within single-seed noise — '
        '<b>no clear optimal k</b>. The "whether" of context matters more than the "how much". '
        '<i>Add a second-seed run under <code>models_seed7/</code> to get error bars.</i>'
    )
st.markdown(
    f'<div class="banner banner-info" style="margin-top: 1rem;">{banner_msg}</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Speaker-policy ablation — static results from the paper (single seed=42 due
# to compute budget; mixed-policy is the configuration the live demo uses).
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Speaker-policy ablation at k = 1 (results from the paper)"):
    sp_data = pd.DataFrame({
        "policy":      ["mixed", "same",  "team",  "enemy", "other"],
        "E precision": [0.850,   0.847,   0.830,   0.874,   0.818 ],
        "E recall":    [0.851,   0.862,   0.871,   0.829,   0.874 ],
        "I precision": [0.910,   0.910,   0.854,   0.938,   0.854 ],
        "I recall":    [0.643,   0.639,   0.665,   0.619,   0.672 ],
        "macro F1":    [0.835,   0.837,   0.837,   0.836,   0.837 ],
    })
    def _highlight_col_max(col):
        is_max = col == col.max() if col.dtype.kind == "f" else [False] * len(col)
        return ["background-color: rgba(180,83,9,0.18); font-weight:700;" if v else ""
                for v in is_max]
    st.dataframe(
        sp_data.style
            .apply(_highlight_col_max, subset=["E precision", "E recall",
                                               "I precision", "I recall"])
            .format({c: "{:.3f}" for c in sp_data.columns if c != "policy"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        '<div class="small-mono" style="margin-top: 0.5rem;">'
        'All 5 policies sit within a <b>0.835–0.840 macro F1 band</b> but occupy '
        'distinct precision–recall operating points. <b>enemy-only</b> context '
        'maximises Implicit precision (0.938); <b>other-speaker</b> context '
        'maximises Implicit recall (0.672). Speaker policy is a free '
        'moderation-threshold knob — invisible at the aggregate metric level. '
        '<br><i>Live demo above uses the <b>mixed</b> policy (all k prior messages).</i>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graceful degradation
# ─────────────────────────────────────────────────────────────────────────────
if not models_found:
    st.error(
        f"No models found under `{MODELS_DIR}/`. Run:\n\n"
        "```\nmkdir -p models && for f in k*_model.tar.gz; do tar -xzf \"$f\" -C models/; done\n```"
    )
    st.stop()

missing = sorted(set(K_VALUES) - set(models_found.keys()))
if missing:
    st.markdown(
        f'<div class="banner banner-warn">Missing k = {missing}. Demo will run with the {len(models_found)} available model(s).</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<h3 style="margin-top: 0;">Settings</h3>', unsafe_allow_html=True)

selected_ks = st.sidebar.multiselect(
    "Compare which k values",
    options=sorted(models_found.keys()),
    default=sorted(models_found.keys()),
    format_func=lambda k: f"k = {k}",
)

device_label = "CUDA" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(
    f'<div class="small-mono" style="margin: 0.5rem 0;">'
    f'<span style="color: var(--c-text-2);">Inference device:</span> '
    f'<b style="color: var(--c-fg);">{device_label}</b>'
    f'</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button("Pre-load all models", use_container_width=True,
                     help="Warm the @st.cache_resource cache. First load of each model is ~5-10s on CPU."):
    bar = st.sidebar.progress(0.0, text="loading…")
    for i, k in enumerate(selected_ks):
        bar.progress(i / max(len(selected_ks), 1), text=f"loading k = {k}…")
        load_model(str(models_found[k]))
    bar.progress(1.0, text="ready")
    time.sleep(0.4)
    bar.empty()
    st.sidebar.success(f"Loaded {len(selected_ks)} model(s).")

st.sidebar.markdown(
    """
    <h3 style="margin-top: 1.5rem;">Labels</h3>
    <div style="font-family: 'Fira Sans', sans-serif; font-size: 0.85rem; line-height: 1.9;">
      <span style="font-family:'Fira Code',monospace; color:#b91c1c; font-weight:700;">E</span>
      <span style="color: var(--c-text-2);"> · Explicit toxicity</span><br>
      <span style="font-family:'Fira Code',monospace; color:#b45309; font-weight:700;">I</span>
      <span style="color: var(--c-text-2);"> · Implicit toxicity</span><br>
      <span style="font-family:'Fira Code',monospace; color:#15803d; font-weight:700;">A</span>
      <span style="color: var(--c-text-2);"> · Action / command</span><br>
      <span style="font-family:'Fira Code',monospace; color:#475569; font-weight:700;">O</span>
      <span style="color: var(--c-text-2);"> · Other (non-toxic)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="small-mono" style="margin-top: 1.5rem; line-height: 1.5;">'
    'First classify is slower while models warm. Subsequent runs are instant (@st.cache_resource).'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state + presets
# ─────────────────────────────────────────────────────────────────────────────
if "rev" not in st.session_state:
    st.session_state.rev = 0
if "ctx_msgs" not in st.session_state:
    st.session_state.ctx_msgs = [(5, "1 15 score lol"), (3, "wow nice")]
if "target_text" not in st.session_state:
    st.session_state.target_text = "ez mid"
if "preset_note" not in st.session_state:
    st.session_state.preset_note = ""


def apply_preset(cfg: dict):
    st.session_state.ctx_msgs = list(cfg["context"])
    st.session_state.target_text = cfg["target"]
    st.session_state.preset_note = cfg["note"]
    st.session_state.rev += 1


st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown("### Examples")

preset_cols = st.columns(len(PRESETS))
for col, (name, cfg) in zip(preset_cols, PRESETS.items()):
    if col.button(name, use_container_width=True, key=f"preset_{name}"):
        apply_preset(cfg)
        st.rerun()

if st.session_state.preset_note:
    st.markdown(
        f'<div class="banner banner-warn" style="margin-top: 0.5rem;">{st.session_state.preset_note}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation builder
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown("### Conversation")
st.markdown(
    '<div style="margin-bottom: 0.75rem;">'
    '<span class="ctx-pill">Context</span> '
    '<span class="small-mono">Prior messages — used by k &gt; 0 models. Oldest first.</span>'
    '</div>',
    unsafe_allow_html=True,
)

rev = st.session_state.rev
if not st.session_state.ctx_msgs:
    st.markdown(
        '<div class="small-mono" style="padding: 8px 12px; background: var(--c-muted); '
        'border-radius: 6px; border: 1px dashed var(--c-border);">'
        '— no context — k &gt; 0 models will see only the target —</div>',
        unsafe_allow_html=True,
    )
else:
    # Mini header row so the slot column isn't a mystery number.
    hdr = st.columns([1, 6, 1])
    hdr[0].markdown(
        '<div class="small-mono" style="text-align:center; letter-spacing:0.08em; '
        'text-transform:uppercase; color:var(--c-text-2);">slot (P<i>n</i>)</div>',
        unsafe_allow_html=True,
    )
    hdr[1].markdown(
        '<div class="small-mono" style="letter-spacing:0.08em; '
        'text-transform:uppercase; color:var(--c-text-2);">message</div>',
        unsafe_allow_html=True,
    )

to_remove = None
for i, (slot, text) in enumerate(st.session_state.ctx_msgs):
    cols = st.columns([1, 6, 1])
    cols[0].number_input(
        f"slot_label_{i}", min_value=0, max_value=9, value=int(slot),
        key=f"slot_{i}_{rev}", label_visibility="collapsed",
        help="Player slot (P0–P9) — which player said this message. The models "
             "were trained with these speaker tags, so changing the slot ID can "
             "shift predictions on borderline inputs.",
    )
    cols[1].text_input(
        f"text_label_{i}", value=text,
        key=f"text_{i}_{rev}", label_visibility="collapsed",
        placeholder=f"prior message #{i+1}",
    )
    if cols[2].button("×", key=f"rm_{i}_{rev}", help="Remove this message"):
        to_remove = i

if to_remove is not None:
    st.session_state.ctx_msgs.pop(to_remove)
    st.session_state.rev += 1
    st.rerun()

btn_cols = st.columns([1.5, 1.5, 6])
if btn_cols[0].button("+ Add prior message", key=f"add_{rev}"):
    st.session_state.ctx_msgs = [
        (int(st.session_state.get(f"slot_{i}_{rev}", s)),
         st.session_state.get(f"text_{i}_{rev}", t))
        for i, (s, t) in enumerate(st.session_state.ctx_msgs)
    ]
    st.session_state.ctx_msgs.append((0, ""))
    st.session_state.rev += 1
    st.rerun()
if btn_cols[1].button("Clear context", key=f"clr_{rev}"):
    st.session_state.ctx_msgs = []
    st.session_state.rev += 1
    st.rerun()

st.markdown(
    '<div style="margin-top: 1.25rem; margin-bottom: 0.5rem;">'
    '<span class="target-pill">Target</span> '
    '<span class="small-mono" style="margin-left: 0.5rem;">— this is what gets classified.</span>'
    '</div>',
    unsafe_allow_html=True,
)
# Use a rev-scoped widget key + explicit value= so target survives every
# rev bump (preset click, add/remove/clear context). The canonical value
# lives in st.session_state.target_text, which we sync back on every run.
target = st.text_input(
    "Target",
    value=st.session_state.target_text,
    key=f"target_input_{rev}",
    label_visibility="collapsed",
    placeholder='e.g. "ez mid"',
)
st.session_state.target_text = target


def collect_ctx() -> List[Tuple[int, str]]:
    rev = st.session_state.rev
    return [
        (int(st.session_state.get(f"slot_{i}_{rev}", s)),
         st.session_state.get(f"text_{i}_{rev}", t))
        for i, (s, t) in enumerate(st.session_state.ctx_msgs)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Classify
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
run = st.button("Classify with all selected models →", type="primary", use_container_width=True)

if run:
    if not target.strip():
        st.warning("Enter a target message.")
        st.stop()
    if not selected_ks:
        st.warning("Select at least one k in the sidebar.")
        st.stop()
    ctx_now = collect_ctx()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### Per-model predictions")

    results_rows = []
    per_model_probs: Dict[int, dict] = {}
    cols = st.columns(len(selected_ks))
    for col_idx, k in enumerate(selected_ks):
        with cols[col_idx]:
            st.markdown(
                f'<div class="small-mono" style="margin-bottom: 0.25rem; letter-spacing: 0.08em; text-transform: uppercase;">MODEL · k = {k}</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("…"):
                tokenizer, model, meta = load_model(str(models_found[k]))
            max_len = meta.get("max_length", 256)
            input_str = format_input(target, ctx_now, k)
            probs, n_tok, truncated = predict(model, tokenizer, input_str, max_len)
            top = max(probs, key=probs.get)
            color = LABEL_COLORS[top]
            st.markdown(
                f"""
                <div class="pred-card" style="border-left-color: {color};">
                  <div class="pred-card-label" style="color: {color};">{top}</div>
                  <div class="pred-card-name">{LABEL_NAMES[top]}</div>
                  <div class="pred-card-conf">{probs[top]*100:.1f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            df = pd.DataFrame({"label": LABELS, "prob": [probs[l] for l in LABELS]})
            bar = px.bar(
                df, x="label", y="prob",
                color="label", color_discrete_map=LABEL_COLORS,
                category_orders={"label": LABELS},
            )
            bar.update_traces(
                hovertemplate="<b>%{x}</b><br>P = %{y:.3f}<extra></extra>",
                marker_line_width=0,
            )
            bar.update_layout(
                showlegend=False, height=180,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#e2e8f0",
                           showline=False, tickfont=dict(family="Fira Code, monospace", size=10, color="#0f172a")),
                xaxis=dict(showline=False, tickfont=dict(family="Fira Code, monospace", size=11, color="#0f172a")),
                xaxis_title=None, yaxis_title=None,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Fira Sans, sans-serif", size=11, color="#0f172a"),
                hoverlabel=dict(font=dict(family="Fira Code, monospace", color="#0f172a"),
                                bgcolor="white", bordercolor="#cbd5e1"),
            )
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False}, key=f"bar_{k}")

            tok_color = "var(--c-danger)" if truncated else "var(--c-text-2)"
            tok_note = " — truncated" if truncated else ""
            st.markdown(
                f'<div class="small-mono">tokens · <b style="color: var(--c-fg);">{n_tok}</b><span style="color: var(--c-text-2);"> / {max_len}</span><span style="color: {tok_color};">{tok_note}</span></div>',
                unsafe_allow_html=True,
            )

            with st.expander(f"Model input (k = {k})"):
                st.code(input_str or "(empty)", language="text")

            per_model_probs[k] = probs
            results_rows.append({
                "k": k, "prediction": top,
                **{f"P({l})": probs[l] for l in LABELS},
                "tokens": n_tok, "truncated": truncated,
            })

    # ─── Agreement / disagreement ───
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    tops = {k: max(p, key=p.get) for k, p in per_model_probs.items()}
    unique_tops = set(tops.values())
    if len(unique_tops) == 1:
        only = next(iter(unique_tops))
        st.markdown(
            f'<div class="banner banner-success">'
            f'<b>All {len(tops)} models agree:</b> '
            f'<span style="font-family: \'Fira Code\', monospace; font-weight: 700; color: {LABEL_COLORS[only]};">{only}</span> '
            f'· {LABEL_NAMES[only]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        breakdown = "  ·  ".join(
            f'<span style="font-family: \'Fira Code\', monospace;">k={k} → <b style="color: {LABEL_COLORS[v]};">{v}</b></span>'
            for k, v in tops.items()
        )
        st.markdown(
            f'<div class="banner banner-warn">'
            f'<b>Disagreement</b> — {len(unique_tops)} distinct top labels &nbsp; {breakdown}<br>'
            f'<span class="small-mono">This is exactly the kind of case the project is about — context flipped the prediction.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ─── Side-by-side ───
    st.markdown("### Side-by-side")
    side_df = pd.DataFrame(results_rows).set_index("k")
    st.dataframe(
        side_df.style.format(
            {**{f"P({l})": "{:.1%}" for l in LABELS}, "tokens": "{:d}"}
        ),
        use_container_width=True,
    )
    csv_buf = io.StringIO()
    side_df.to_csv(csv_buf)
    st.download_button(
        "Download CSV",
        csv_buf.getvalue(),
        file_name=f"conda_compare_{int(time.time())}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation metrics — Plotly heatmap (replaces background_gradient — no matplotlib needed)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Validation metrics — multi-seed aggregate"):
    # Aggregate F1 across all seed folders (mean per (k, metric)) and overlay
    # std (n≥3) or half-range (n=2) so the heatmap cells convey variance too.
    multi = read_multi_seed_metas()
    metric_keys = [("F1 macro", "eval_f1_macro"),
                   ("F1 (E)",   "eval_f1_E"),
                   ("F1 (I)",   "eval_f1_I"),
                   ("F1 (A)",   "eval_f1_A"),
                   ("F1 (O)",   "eval_f1_O"),
                   ("Accuracy", "eval_accuracy")]
    # Build a per-k dict of {metric_label: [values across seeds]}
    by_k = {}
    for _, per_k in multi.items():
        for k, m in per_k.items():
            em = m.get("eval_metrics", {})
            slot = by_k.setdefault(k, {col: [] for col, _ in metric_keys})
            for col, key in metric_keys:
                v = em.get(key)
                if v is not None:
                    slot[col].append(float(v))

    if by_k:
        ks_sorted = sorted(by_k.keys())
        cols = [c for c, _ in metric_keys]
        mean_grid, text_grid = [], []
        for k in ks_sorted:
            mean_row, text_row = [], []
            for c in cols:
                vals = by_k[k][c]
                if not vals:
                    mean_row.append(None); text_row.append("—"); continue
                mean = sum(vals) / len(vals)
                mean_row.append(mean)
                if len(vals) == 1:
                    text_row.append(f"{mean:.4f}")
                elif len(vals) == 2:
                    half = (max(vals) - min(vals)) / 2
                    text_row.append(f"{mean:.3f}<br>±{half:.3f}")
                else:
                    import statistics
                    std = statistics.stdev(vals)
                    text_row.append(f"{mean:.3f}<br>±{std:.3f}")
            mean_grid.append(mean_row)
            text_grid.append(text_row)

        n_seeds_per_k = {k: max(len(by_k[k][c]) for c in cols) for k in ks_sorted}
        y_labels = [f"k = {k}  (n={n_seeds_per_k[k]})" for k in ks_sorted]

        heat = go.Figure(data=go.Heatmap(
            z=mean_grid,
            x=cols,
            y=y_labels,
            colorscale=[
                [0.0, "#fee2e2"],
                [0.5, "#fef3c7"],
                [1.0, "#dcfce7"],
            ],
            zmin=0.70, zmax=0.96,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(family="Fira Code, monospace", size=11, color="#0f172a"),
            colorbar=dict(
                title=dict(text="F1<br>(mean)", font=dict(family="Fira Sans, sans-serif", color="#0f172a")),
                tickfont=dict(family="Fira Code, monospace", size=10, color="#0f172a"),
                len=0.7,
            ),
            hovertemplate="<b>%{y}</b> · %{x}<br>mean = %{z:.4f}<extra></extra>",
            xgap=2, ygap=2,
        ))
        heat.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=10),
            font=dict(family="Fira Sans, sans-serif", color="#0f172a"),
            xaxis=dict(side="top", tickfont=dict(family="Fira Code, monospace", size=11, color="#0f172a")),
            yaxis=dict(tickfont=dict(family="Fira Code, monospace", size=11, color="#0f172a"), autorange="reversed"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hoverlabel=dict(font=dict(family="Fira Code, monospace", color="#0f172a"),
                            bgcolor="white", bordercolor="#cbd5e1"),
        )
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})

        seed_labels = sorted({s for s in multi.keys()})
        st.markdown(
            f'<div class="small-mono" style="margin-top: 0.5rem;">'
            f'Mean ± std across <b>{len(seed_labels)} seed(s)</b> '
            f'({", ".join(seed_labels)}). '
            'Notice how the spread (the ± value, and the cell color steadiness) '
            'grows from k=1 to k=10 — context-aware models become increasingly '
            'sensitive to random initialization as the input window widens.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No meta.json files found in any seed folder.")
