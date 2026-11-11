"""P7-B: GPT-2 perplexity + GLTR text detector.

Usage:
    from P7_B_predict import predict, score_texts
    print(predict('Artificial intelligence is changing education.'))

Downloads the trained classifier from captainfan/p7-gpt2-gltr-detector.
GPT-2 and the classifier are cached locally after the first download.
Scores are model estimates, not proof of authorship.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from huggingface_hub import hf_hub_download

MODEL_NAME = os.environ.get("P7_B_MODEL", "gpt2")
CALIBRATOR_REPO = os.environ.get("P7_B_CALIBRATOR_REPO", "captainfan/p7-gpt2-gltr-detector")
CALIBRATOR_FILE = "P7_B_calib.pkl"
# Optional: set P7_B_CALIBRATOR to a local .pkl path to work offline.
LOCAL_CALIBRATOR = os.environ.get("P7_B_CALIBRATOR")
MAX_LEN = 256  # Same as Section 7 of P7_B.ipynb
THRESHOLD = 0.5
FEATURE_COLUMNS = ["log_ppl", "gltr_top10", "gltr_top100", "gltr_top1000", "gltr_above1000"]
_CACHE = {}


def load_model():
    """Load GPT-2 and the fitted calibration pipeline once per process."""
    if "components" not in _CACHE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()
        if LOCAL_CALIBRATOR:
            calibrator_path = Path(LOCAL_CALIBRATOR)
            if not calibrator_path.is_file():
                raise FileNotFoundError(f"Classifier not found: {calibrator_path}")
        else:
            calibrator_path = hf_hub_download(
                repo_id=CALIBRATOR_REPO,
                filename=CALIBRATOR_FILE,
            )
        # Only load .pkl files from repositories you trust.
        calibrator = joblib.load(calibrator_path)
        _CACHE["components"] = (tokenizer, model, calibrator, device)
    return _CACHE["components"]


def extract_features(text: str) -> dict[str, float]:
    """Exactly the five features used in the notebook's Section 7."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    tokenizer, model, _, device = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError("Text must contain at least two GPT-2 tokens.")

    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        predictions = logits[:-1]
        targets = input_ids[0, 1:]
        log_probs = torch.log_softmax(predictions, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_ppl = -target_log_probs.mean().item()
        target_logits = predictions.gather(1, targets.unsqueeze(1))
        ranks = ((predictions > target_logits).sum(dim=1) + 1).cpu().numpy()

    return {
        "log_ppl": log_ppl,
        "gltr_top10": float(np.mean(ranks <= 10)),
        "gltr_top100": float(np.mean((ranks > 10) & (ranks <= 100))),
        "gltr_top1000": float(np.mean((ranks > 100) & (ranks <= 1000))),
        "gltr_above1000": float(np.mean(ranks > 1000)),
    }


def predict(text: str) -> dict:
    """Return the shared group format: prob_ai, label, extra."""
    features = extract_features(text)
    tokenizer, _, calibrator, _ = load_model()
    # The notebook trained on a NumPy array in this exact column order.
    x = np.array([[features[name] for name in FEATURE_COLUMNS]], dtype=float)
    prob_ai = float(calibrator.predict_proba(x)[0, 1])
    truncated = len(tokenizer(text, truncation=False)["input_ids"]) > MAX_LEN
    return {
        "prob_ai": round(prob_ai, 4),
        "label": "AI" if prob_ai >= THRESHOLD else "Human",
        "extra": {
            "truncated": truncated,
            "ppl": round(math.exp(features["log_ppl"]), 4),
            **features,
        },
    }


def score_texts(texts: list[str], batch_size: int = 32) -> list[float]:
    """Return AI scores in input order; batch_size controls progress chunks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    scores = []
    for start in range(0, len(texts), batch_size):
        for text in texts[start:start + batch_size]:
            scores.append(predict(text)["prob_ai"])
    return scores


if __name__ == "__main__":
    import json
    sample = "Artificial intelligence has the potential to transform education by providing personalized learning experiences."
    print(json.dumps(predict(sample), indent=2, ensure_ascii=False))
