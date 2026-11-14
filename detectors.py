"""Loads the four detectors (A–D) and normalises their outputs.

Each member uploaded P7_X_predict.py with predict(text) -> dict.
If a file fails to import, a mock detector is used so the app still runs.
Field names differ a little between members; normalise() maps them to
one shape for the UI.
"""
import glob
import hashlib
import importlib
import os

import streamlit as st

DETECTORS = {
    "A": ("Stylometric + XGBoost", "P7_A_predict"),
    "B": ("GPT-2 perplexity + GLTR", "P7_B_predict"),
    "C": ("Fine-tuned RoBERTa", "P7_C_predict"),
    "D": ("LLM judge (DeepSeek)", "P7_D_predict"),
}


def _export_secrets():
    """Copy top-level secrets (e.g. DEEPSEEK_API_KEY) into environment variables."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ.setdefault(k, v)
    except Exception:
        pass


def _point_d_to_fewshot():
    """D's predict() reads data/fewshot_examples.json; D uploaded it under D_others/, so pass that path."""
    if os.environ.get("P7_D_FEWSHOT"):
        return
    hits = sorted(glob.glob("D_others/**/fewshot_examples.json", recursive=True)) + glob.glob("data/fewshot_examples.json")
    if hits:
        os.environ["P7_D_FEWSHOT"] = os.path.abspath(hits[0])


def _mock(key):
    def predict(text: str) -> dict:
        h = int(hashlib.md5((key + text).encode()).hexdigest(), 16)
        p = (h % 1000) / 1000
        extra = {}
        if key == "B":
            extra = {"ppl": round(5 + (1 - p) * 40, 1)}
        if key == "D":
            extra = {"reason": "Mock reason: uniform sentence length and generic phrasing."}
        return {"prob_ai": p, "label": "AI" if p >= 0.5 else "Human", "extra": extra}

    return predict


@st.cache_resource(show_spinner="Loading detectors…")
def load_detectors() -> dict:
    _export_secrets()
    _point_d_to_fewshot()
    out = {}
    for key, (name, module) in DETECTORS.items():
        try:
            m = importlib.import_module(module)
            out[key] = {"name": name, "predict": m.predict, "mock": False, "error": None}
        except Exception as e:  # missing file, missing package…
            out[key] = {"name": name, "predict": _mock(key), "mock": True, "error": f"{type(e).__name__}: {e}"}
    return out


def normalise(key: str, r):
    """Map each member's output to: prob_ai, label, usable, notes, perplexity, reason, sentences."""
    if not r or "prob_ai" not in r:
        return None
    extra = r.get("extra") or {}
    out = {
        "prob_ai": min(max(float(r["prob_ai"]), 0.0), 1.0),
        "label": r.get("label", ""),
        "usable": True,     # counts towards the overall verdict
        "notes": [],
        "perplexity": extra.get("perplexity", extra.get("ppl")),   # B uses "ppl"
        "reason": extra.get("reason"),                               # D
        "sentences": extra.get("sentences"),                         # C: [{"text", "prob_ai"}]
    }
    if extra.get("note"):                       # A: text too short -> neutral 0.5
        out["notes"].append(extra["note"])
        out["usable"] = False
    if extra.get("truncated"):                  # B, C: only the first 256 tokens are scored
        out["notes"].append("only the first 256 tokens were scored")
    if extra.get("tfidf_lr_prob_ai") is not None:
        out["notes"].append(f"TF-IDF + LR baseline: {extra['tfidf_lr_prob_ai']:.2f}")
    if extra.get("score_type"):                 # D: LLM score, not calibrated
        out["notes"].append("LLM score, not a calibrated probability")
    return out


def run(key: str, detector: dict, text: str):
    """Call a detector safely. Returns the normalised result, or None if it failed."""
    try:
        return normalise(key, detector["predict"](text))
    except Exception:
        return None
