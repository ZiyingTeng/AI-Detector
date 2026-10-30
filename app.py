"""AI Text Detector — Streamlit front end (F).

Run:  streamlit run app.py
"""
import os
import re

import pandas as pd
import streamlit as st

import gltr
from detectors import load_detectors, run

st.set_page_config(page_title="AI Text Detector", layout="wide")

SAMPLES = {
    "Human-written (sample)": "honestly i didn't expect the exam to be that long. spent way too much time on q3 and "
    "had to rush the last part, so we'll see. anyway, going for bubble tea later if anyone's free?",
    "ChatGPT-style (sample)": "Artificial intelligence has the potential to transform education by providing "
    "personalized learning experiences. However, it is important to consider the ethical implications, including "
    "privacy concerns and the risk of over-reliance on technology.",
}


def load_samples() -> dict:
    path = "P7_demo_texts.csv"  # uploaded by D: columns name,text
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return dict(zip(df["name"], df["text"]))
        except Exception:
            pass
    return SAMPLES


detectors = load_detectors()
samples = load_samples()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Detectors")
    enabled = {}
    for key, d in detectors.items():
        label = f"{key} · {d['name']}" + ("  (mock)" if d["mock"] else "")
        enabled[key] = st.checkbox(label, value=True, key=f"use_{key}")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    sample_name = st.selectbox("Example text", ["(type your own)"] + list(samples))
    if any(d["mock"] for d in detectors.values()):
        st.caption("Detectors marked (mock) return random scores until their predict file is uploaded.")

# ---------------- main ----------------
st.title("AI Text Detector")
default_text = samples.get(sample_name, "")
text = st.text_area("Paste a passage", value=default_text, height=160, key=f"text_{sample_name}")
go = st.button("Detect", type="primary", disabled=not text.strip())

if go and text.strip():
    results = {}
    with st.spinner("Running detectors…"):
        for key, d in detectors.items():
            if enabled[key]:
                results[key] = run(d, text)

    valid = {k: r for k, r in results.items() if r}
    # ---- overall verdict ----
    if valid:
        mean_p = sum(r["prob_ai"] for r in valid.values()) / len(valid)
        n_ai = sum(r["prob_ai"] >= threshold for r in valid.values())
        if abs(mean_p - threshold) < 0.1:
            verdict = "Uncertain"
        elif mean_p >= threshold:
            verdict = "Likely AI-generated"
        else:
            verdict = "Likely human-written"
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", verdict)
        c2.metric("Mean AI probability", f"{mean_p:.2f}")
        c3.metric("Detectors flagging AI", f"{n_ai} / {len(valid)}")
    st.caption("Detector outputs are probabilistic and should not be used as the sole evidence of misconduct.")

    # ---- per-detector results ----
    st.subheader("Detectors")
    for key, r in results.items():
        d = detectors[key]
        name = f"**{key} · {d['name']}**" + (" *(mock)*" if d["mock"] else "")
        cols = st.columns([2, 5, 1])
        cols[0].markdown(name)
        if r is None:
            cols[1].warning("Unavailable (call failed)")
            continue
        cols[1].progress(min(max(r["prob_ai"], 0.0), 1.0))
        cols[2].markdown(f"`{r['prob_ai']:.2f}`")
        extra = r.get("extra") or {}
        if "perplexity" in extra:
            st.caption(f"Perplexity: {extra['perplexity']}")
        if "reason" in extra:
            st.caption(f"LLM reasoning: {extra['reason']}")

    # ---- GLTR colouring ----
    st.subheader("Token ranks (GLTR)")
    pairs, gltr_mock = gltr.token_ranks(text)
    if gltr_mock:
        st.caption("GPT-2 not installed: showing random colours. Install torch and transformers for real ranks.")
    left, right = st.columns([3, 1])
    left.markdown(gltr.render(pairs), unsafe_allow_html=True)
    left.markdown(gltr.legend(), unsafe_allow_html=True)
    frac = gltr.fractions(pairs)
    right.bar_chart(pd.Series(frac, name="share"), height=220)

    # ---- sentence level ----
    with st.expander("Sentence-level view"):
        pick = next((k for k in ["C", "A", "B"] if k in valid), None)
        if pick:
            st.caption(f"Each sentence scored by {pick} · {detectors[pick]['name']}")
            sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
            parts = []
            for s in sents:
                r = run(detectors[pick], s)
                p = r["prob_ai"] if r else 0
                parts.append(
                    f'<span style="background:rgba(244,67,54,{0.08 + 0.5 * p:.2f});border-radius:3px;'
                    f'padding:1px 2px" title="{p:.2f}">{s}</span>'
                )
            st.markdown('<div style="line-height:2">' + " ".join(parts) + "</div>", unsafe_allow_html=True)
        else:
            st.info("Enable A, B or C to see sentence-level scores.")
