"""Side-by-side demo: the same AI text before and after a paraphrase attack."""
import os

import pandas as pd
import streamlit as st

from detectors import load_detectors, run
from results_data import load_pairs

st.set_page_config(page_title="Attack demo · AI Text Detector", layout="wide")
st.title("Attack demo")
st.caption("The same AI-generated passage, before and after a paraphrase attack, scored by every detector.")

pairs = load_pairs()
if pairs is None and os.path.exists("P7_demo_texts.csv"):
    # fallback: D's demo texts contain a newer-LLM passage and its paraphrased version
    demo = pd.read_csv("P7_demo_texts.csv", encoding="utf-8-sig").set_index("name")["text"]
    if {"new_model", "attacked"} <= set(demo.index):
        pairs = pd.DataFrame([{"id": 1, "original": demo["new_model"], "attacked": demo["attacked"],
                               "attack_type": "paraphrase"}])
if pairs is None:
    st.info("Waiting for attack pairs (results/attack_pairs.csv).")
    st.stop()

if "attack_type" not in pairs:
    pairs["attack_type"] = "attack"
if "id" not in pairs:
    pairs["id"] = range(1, len(pairs) + 1)

detectors = load_detectors()
labels = [f"#{r.id} · {r.attack_type}" for r in pairs.itertuples()]
i = st.selectbox("Pair", range(len(pairs)), format_func=lambda k: labels[k])
row = pairs.iloc[i]

if st.button("Score both", type="primary"):
    left, right = st.columns(2)
    for col, title, text in [(left, "Original (AI-generated)", row["original"]),
                             (right, f"After attack: {row['attack_type']}", row["attacked"])]:
        with col:
            st.subheader(title)
            st.write(text)
            for key, d in detectors.items():
                r = run(key, d, text)
                name = f"{key} · {d['name']}" + (" (mock)" if d["mock"] else "")
                if r is None:
                    st.caption(f"{name}: unavailable")
                else:
                    st.progress(r["prob_ai"], text=f"{name}: {r['prob_ai']:.2f}")
else:
    left, right = st.columns(2)
    left.subheader("Original (AI-generated)")
    left.write(row["original"])
    right.subheader(f"After attack: {row['attack_type']}")
    right.write(row["attacked"])
