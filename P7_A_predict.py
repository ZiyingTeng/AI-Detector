"""Detector A: stylometric features + XGBoost."""
import math
import os
import re

import numpy as np

TRANSITIONS = {
    "however", "moreover", "additionally", "furthermore", "therefore", "thus", "consequently",
    "overall", "ultimately", "nevertheless", "similarly", "specifically", "importantly", "generally",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
HEDGES = {"may", "might", "can", "could", "often", "typically", "generally", "various", "several"}
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "is", "that", "it", "for", "as", "with", "this", "are",
    "be", "by", "or", "on", "not", "which", "can", "but", "you", "so",
]
FUNCTION_SET = set(FUNCTION_WORDS) | {
    "an", "at", "from", "was", "were", "have", "has", "had", "will", "would", "if", "than", "then",
    "there", "their", "they", "he", "she", "his", "her", "its", "into", "also", "such", "these", "those",
}

# sentences end at . ! ? only; line breaks are treated as spaces
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

FEATURE_NAMES = [
    "sent_len_mean", "sent_len_std", "sent_len_cv", "sent_len_max",
    "word_len_mean", "word_len_std",
    "mattr", "hapax_ratio", "repeat_bigram_ratio",
    "comma_per_word", "period_per_word", "semicolon_colon_per_word", "question_excl_per_word",
    "quote_paren_dash_per_word", "digit_char_ratio", "upper_char_ratio",
    "transition_ratio", "first_person_ratio", "contraction_ratio", "function_word_ratio", "hedge_ratio",
] + [f"fw_{w}" for w in FUNCTION_WORDS]

# every text is cut to its first WINDOW_WORDS words so that all texts have the same length
WINDOW_WORDS = 60
MIN_WORDS = 40


def truncate_words(text: str, n: int = WINDOW_WORDS) -> str:
    ms = list(re.finditer(r"\S+", text))
    return text if len(ms) <= n else text[:ms[n - 1].end()]


def _mattr(words, window=30):
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    vals = [len(set(words[i:i + window])) / window for i in range(0, n - window + 1, 5)]
    return float(np.mean(vals))


def extract_features(text: str) -> list:
    text = truncate_words(text if isinstance(text, str) else "")
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s for s in SENT_SPLIT.split(text) if s and s.strip()]
    words = [w.lower() for w in WORD_RE.findall(text)]
    n = max(len(words), 1)
    nchar = max(len(text), 1)

    sent_lens = np.array([len(WORD_RE.findall(s)) for s in sents] or [0], dtype=float)
    s_mean = float(sent_lens.mean())
    s_std = float(sent_lens.std()) if len(sent_lens) > 1 else 0.0
    wl = np.array([len(w) for w in words], dtype=float) if words else np.array([0.0])

    bigrams = list(zip(words, words[1:]))
    rep_bigram = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1) / n

    feats = [
        s_mean, s_std, (s_std / s_mean if s_mean > 0 else 0.0), float(sent_lens.max()),
        float(wl.mean()), float(wl.std()),
        _mattr(words), hapax, rep_bigram,
        text.count(",") / n, text.count(".") / n, (text.count(";") + text.count(":")) / n,
        (text.count("?") + text.count("!")) / n,
        sum(text.count(c) for c in "\"'()-—") / n,
        sum(c.isdigit() for c in text) / nchar, sum(c.isupper() for c in text) / nchar,
        sum(w in TRANSITIONS for w in words) / n,
        sum(w in FIRST_PERSON for w in words) / n,
        sum("'" in w for w in words) / n,
        sum(w in FUNCTION_SET for w in words) / n,
        sum(w in HEDGES for w in words) / n,
    ] + [counts.get(w, 0) / n for w in FUNCTION_WORDS]

    return [float(x) if math.isfinite(x) else 0.0 for x in feats]


def featurize_many(texts) -> np.ndarray:
    return np.array([extract_features(t) for t in texts], dtype=float)


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "P7_A_model.pkl")
_BUNDLE = None


def _load():
    # pickle holds {"xgb": classifier, "threshold": float, "tfidf_lr": pipeline}; loaded on first call
    global _BUNDLE
    if _BUNDLE is None:
        import joblib

        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def predict(text: str) -> dict:
    words = WORD_RE.findall(text if isinstance(text, str) else "")
    if len(words) < MIN_WORDS:
        return {"prob_ai": 0.5, "label": "Human",
                "extra": {"note": f"text too short (at least {MIN_WORDS} words needed)"}}

    b = _load()
    text = truncate_words(text)
    p = float(b["xgb"].predict_proba(np.array([extract_features(text)]))[0][1])
    p = min(max(p, 0.0), 1.0)
    thr = float(b.get("threshold", 0.5))
    extra = {}
    if b.get("tfidf_lr") is not None:
        extra["tfidf_lr_prob_ai"] = round(float(b["tfidf_lr"].predict_proba([text])[0][1]), 3)
    return {"prob_ai": p, "label": "AI" if p >= thr else "Human", "extra": extra}
