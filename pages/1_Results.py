"""Experiment results page."""
import altair as alt
import pandas as pd
import streamlit as st

from results_data import DATASET_ORDER, load_attack, load_auroc, load_findings

st.set_page_config(page_title="Results · AI Text Detector", layout="wide")
st.title("Experiment results")

auroc, src = load_auroc()
attack = load_attack()
findings = load_findings()

# ---- AUROC table ----
st.subheader("Detector performance by test set")
if auroc is None:
    st.info("No result files found yet.")
else:
    st.caption(f"Source: {src}")
    metrics = [m for m in ["auroc", "f1", "tpr_at_1fpr"] if m in auroc]
    names = {"auroc": "AUROC", "f1": "F1", "tpr_at_1fpr": "TPR@1%FPR"}
    metric = st.radio("Metric", metrics, format_func=names.get, horizontal=True)
    table = auroc.pivot_table(index="dataset", columns="detector", values=metric)
    order = [d for d in DATASET_ORDER if d in table.index] + [d for d in table.index if d not in DATASET_ORDER]
    table = table.reindex(order).astype(float)
    st.dataframe(
        table.style.format("{:.2f}", na_rep="–").highlight_max(axis=1, props="font-weight:bold"),
        use_container_width=True,
    )
    st.caption("Bold = best detector on each test set. – = not reported yet.")

    long = table.reset_index().melt(id_vars="dataset", var_name="detector", value_name=names[metric]).dropna()
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("dataset:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="detector:N",
            y=alt.Y(f"{names[metric]}:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("detector:N", title="Detector"),
            tooltip=["dataset", "detector", alt.Tooltip(f"{names[metric]}:Q", format=".2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

# ---- attack before / after ----
st.subheader("Before and after paraphrase attacks")
if attack is None:
    st.info("Waiting for E's attack results (results/attack.csv).")
else:
    long = attack.melt(id_vars="detector", value_vars=["auroc_before", "auroc_after"],
                       var_name="condition", value_name="AUROC")
    long["condition"] = long["condition"].map({"auroc_before": "Before", "auroc_after": "After attack"})
    chart = (
        alt.Chart(long)
        .mark_bar(clip=True)
        .encode(
            x=alt.X("detector:N", title="Detector"),
            xOffset=alt.XOffset("condition:N", sort=["Before", "After attack"]),
            y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0.25, 1])),
            color=alt.Color("condition:N", title=None, sort=["Before", "After attack"]),
            tooltip=["detector", "condition", alt.Tooltip("AUROC:Q", format=".2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

# ---- findings ----
st.subheader("Key findings")
if findings:
    for f in findings:
        st.markdown(f"- {f}")
else:
    st.caption("Waiting for E's key findings (results/findings.txt).")
