"""Score D's new-generator test set (P7_new_generators.csv) with detectors B and C,
and append the results to B's and C's result files for E.

Run from the repo root:
    python tools/run_new_generators.py            # both B and C
    python tools/run_new_generators.py --only C   # just one

No API key is needed: B uses GPT-2 and C uses RoBERTa, both run locally
(first run downloads them from Hugging Face).
Running it again replaces the earlier "New generators" rows, so it is safe to rerun.
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "P7_new_generators.csv")
DATASET = "New generators"   # same name A used
THRESHOLD = 0.5

# each member's own conventions, so the new rows match their existing rows
DETECTORS = {
    "B": {"module": "P7_B_predict", "method": "gpt2-gltr-lr", "config": "train=HC3",
          "f1": "binary", "decimals": None},
    "C": {"module": "P7_C_predict", "method": "roberta-base-ft", "config": "train=ALL",
          "f1": "macro", "decimals": 2},
}


def tpr_at_fpr(y, s, target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def replace_rows(path, new_rows):
    """Drop earlier New-generators rows from a member's CSV, then append the new ones."""
    old = pd.read_csv(path, encoding="utf-8-sig")
    old = old[old["dataset"].astype(str).str.strip().str.lower() != DATASET.lower()]
    out = pd.concat([old, pd.DataFrame(new_rows)[old.columns]], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8")


def run(key):
    cfg = DETECTORS[key]
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    texts = df["text"].astype(str).tolist()
    y = (df["label"].astype(str).str.strip().str.upper() == "AI").astype(int).to_numpy()
    print(f"[{key}] scoring {len(texts)} texts ({y.sum()} AI, {len(y) - y.sum()} human)…")

    t0 = time.time()
    mod = importlib.import_module(cfg["module"])
    scores = np.asarray(mod.score_texts(texts), dtype=float)
    pred = (scores >= THRESHOLD).astype(int)
    print(f"[{key}] done in {time.time() - t0:.0f} s")

    auroc = roc_auc_score(y, scores)
    f1 = f1_score(y, pred, average=cfg["f1"], zero_division=0)
    tpr = tpr_at_fpr(y, scores)
    d = cfg["decimals"]
    fmt = (lambda v: round(v, d)) if d is not None else (lambda v: v)

    folder = os.path.join(ROOT, f"{key}_others", f"{key}_to_E")
    replace_rows(os.path.join(folder, f"P7_{key}_auroc.csv"),
                 [{"dataset": DATASET, "detector": key, "auroc": fmt(auroc), "f1": fmt(f1), "tpr_at_1fpr": fmt(tpr)}])
    replace_rows(os.path.join(folder, f"P7_{key}_metrics.csv"),
                 [{"dataset": DATASET, "method": cfg["method"], "config": cfg["config"], "metric": m, "value": v}
                  for m, v in [("AUROC", auroc), ("F1", f1), ("TPR@1%FPR", tpr)]])
    replace_rows(os.path.join(folder, f"P7_{key}_preds.csv"),
                 [{"dataset": DATASET, "id": i, "gold": g, "pred": p, "score": s}
                  for i, g, p, s in zip(df["uid"], y, pred, scores)])

    f1_name = "macro-F1" if cfg["f1"] == "macro" else "F1"
    print(f"[{key}] {DATASET}: AUROC {auroc:.4f} · {f1_name} {f1:.4f} · TPR@1%FPR {tpr:.4f}")
    print(f"[{key}] appended to {folder}/P7_{key}_auroc.csv, _metrics.csv, _preds.csv\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DETECTORS), help="run just one detector")
    args = ap.parse_args()
    for k in ([args.only] if args.only else list(DETECTORS)):
        run(k)
