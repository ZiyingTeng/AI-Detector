"""Finds the result files wherever members put them.

Priority: results/ (agreed location) → E_others/ (E's own folder) → each
member's X_others/X_to_E/ files. Column names and dataset names are
normalised so the pages can treat them the same.
"""
import glob
import os

import pandas as pd

DATASET_ORDER = ["HC3", "MAGE", "RAID", "New generators", "Attacked"]
DATASET_ALIASES = {
    "hc3": "HC3",
    "mage": "MAGE",
    "raid": "RAID",
    "raid (subset)": "RAID",
    "raid subset": "RAID",
    "new generators": "New generators",
    "new generator": "New generators",
    "new_generators": "New generators",
    "attacked": "Attacked",
    "attack": "Attacked",
    "after attack": "Attacked",
    "paraphrased": "Attacked",
}


def _first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _read(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _norm_dataset(name):
    s = str(name).strip()
    return DATASET_ALIASES.get(s.lower(), s)


def load_auroc():
    """Returns (dataframe, source description) or (None, None)."""
    path = _first(["results/auroc.csv", "results/*auroc*.csv", "E_others/**/*auroc*.csv"])
    if path:
        df, src = _read(path), path
    else:
        parts = sorted(glob.glob("*_others/*_to_E/P7_*_auroc.csv"))
        if not parts:
            return None, None
        df = pd.concat([_read(p) for p in parts], ignore_index=True)
        src = "each member's own results (E's summary not uploaded yet)"
    df["dataset"] = df["dataset"].map(_norm_dataset)
    df["detector"] = df["detector"].astype(str).str.strip()
    for col in ["auroc", "f1", "tpr_at_1fpr"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, src


def load_attack():
    path = _first(["results/attack.csv", "results/*attack*.csv", "E_others/**/*attack*.csv"])
    path = path if path and "pair" not in os.path.basename(path).lower() else None
    if not path:
        return None
    df = _read(path)
    return df if {"detector", "auroc_before", "auroc_after"} <= set(df.columns) else None


def load_pairs():
    path = _first(["results/attack_pairs.csv", "results/*pair*.csv", "E_others/**/*pair*.csv"])
    if not path:
        return None
    df = _read(path)
    return df if {"original", "attacked"} <= set(df.columns) else None


def load_findings():
    path = _first(["results/findings.txt", "results/*finding*.txt", "E_others/**/*finding*.txt"])
    if not path:
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [l.strip().lstrip("-• ").strip() for l in f if l.strip()]
