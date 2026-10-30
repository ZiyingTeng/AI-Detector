"""Loads the four detectors (A–D).

Each member uploads P7_X_predict.py with a predict(text) -> dict function.
If a file is missing or fails to import, a mock detector is used instead,
so the app always runs.
"""
import hashlib
import importlib
import os

import streamlit as st

DETECTORS = {
    "A": ("Stylometric + XGBoost", "P7_A_predict"),
    "B": ("Perplexity / GLTR", "P7_B_predict"),
    "C": ("Fine-tuned classifier", "P7_C_predict"),
    "D": ("LLM judge", "P7_D_predict"),
}


def _export_secrets():
    """Copy top-level secrets (e.g. OPENAI_API_KEY) into environment variables."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ.setdefault(k, v)
    except Exception:
        pass


def _mock(key):
    def predict(text: str) -> dict:
        h = int(hashlib.md5((key + text).encode()).hexdigest(), 16)
        p = (h % 1000) / 1000
        extra = {}
        if key == "B":
            extra = {"perplexity": round(5 + (1 - p) * 40, 1)}
        if key == "D":
            extra = {"reason": "Mock reason: uniform sentence length and generic phrasing."}
        return {"prob_ai": p, "label": "AI" if p >= 0.5 else "Human", "extra": extra}

    return predict


@st.cache_resource(show_spinner="Loading detectors…")
def load_detectors() -> dict:
    _export_secrets()
    out = {}
    for key, (name, module) in DETECTORS.items():
        try:
            m = importlib.import_module(module)
            out[key] = {"name": name, "predict": m.predict, "mock": False}
        except Exception as e:  # missing file, missing model, missing package…
            out[key] = {"name": name, "predict": _mock(key), "mock": True, "error": repr(e)}
    return out


def run(detector: dict, text: str):
    """Call a detector safely. Returns the result dict, or None if it failed."""
    try:
        return detector["predict"](text)
    except Exception:
        return None
