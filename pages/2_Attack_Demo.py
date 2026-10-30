"""Side-by-side demo: the same AI text before and after a paraphrase attack."""
import os

import pandas as pd
import streamlit as st

from detectors import load_detectors, run

st.set_page_config(page_title="Attack demo · AI Text Detector", layout="wide")
st.title("Attack demo")

PAIRS = "results/attack_pairs.csv"  # id,original,attacked,attack_type

if os.path.exists(PAIRS):
    pairs = pd.read_csv(PAIRS)
else:
    st.info("Showing a sample pair until E uploads results/attack_pairs.csv.")
    pairs = pd.DataFrame([{
        "id": 1,
        "original": "Artificial intelligence has the potential to transform education by providing personalized "
                    "learning experiences. However, it is important to consider the ethical implications.",
        "attacked": "AI could really shake up how we learn, giving each student lessons that fit them. Still, "
                    "we can't ignore the ethics side of it.",
        "attack_type": "paraphrase",
    }])

detectors = load_detectors()
labels = [f"#{r.id} · {r.attack_type}" for r in pairs.itertuples()]
i = st.selectbox("Pair", range(len(pairs)), format_func=lambda k: labels[k])
row = pairs.iloc[i]

left, right = st.columns(2)
for col, title, text in [(left, "Original (AI-generated)", row["original"]),
                         (right, f"After attack: {row['attack_type']}", row["attacked"])]:
    with col:
        st.subheader(title)
        st.write(text)
        for key, d in detectors.items():
            r = run(d, text)
            name = f"{key} · {d['name']}" + (" (mock)" if d["mock"] else "")
            if r is None:
                st.caption(f"{name}: unavailable")
            else:
                st.progress(min(max(r["prob_ai"], 0.0), 1.0), text=f"{name}: {r['prob_ai']:.2f}")
