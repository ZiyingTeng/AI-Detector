"""Experiment results page. Reads E's files in results/; shows sample data until they exist."""
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Results · AI Text Detector", layout="wide")
st.title("Experiment results")

AUROC = "results/auroc.csv"            # dataset,detector,auroc,f1,tpr_at_1fpr
ATTACK = "results/attack.csv"          # detector,auroc_before,auroc_after
FINDINGS = "results/findings.txt"      # one finding per line

sample = False
if os.path.exists(AUROC):
    auroc = pd.read_csv(AUROC)
else:
    sample = True
    rows = []
    for ds, vals in {"HC3": [.91, .86, .99, .72], "MAGE": [.84, .81, .93, .66],
                     "RAID (subset)": [.78, .80, .88, .63], "New generators": [.70, .79, .74, .61]}.items():
        for det, v in zip("ABCD", vals):
            rows.append({"dataset": ds, "detector": det, "auroc": v, "f1": v - .05, "tpr_at_1fpr": v - .45})
    auroc = pd.DataFrame(rows)

if os.path.exists(ATTACK):
    attack = pd.read_csv(ATTACK)
else:
    sample = True
    attack = pd.DataFrame({"detector": list("ABCD"), "auroc_before": [.84, .81, .93, .66],
                           "auroc_after": [.62, .58, .69, .60]})

findings = []
if os.path.exists(FINDINGS):
    findings = [l.strip() for l in open(FINDINGS, encoding="utf-8") if l.strip()]

if sample:
    st.info("Showing sample numbers until E uploads files to results/.")

# ---- AUROC table ----
st.subheader("AUROC by dataset and detector")
metric = st.radio("Metric", ["auroc", "f1", "tpr_at_1fpr"], horizontal=True)
table = auroc.pivot_table(index="dataset", columns="detector", values=metric)
st.dataframe(table.style.format("{:.2f}").highlight_max(axis=1, props="font-weight:bold"), use_container_width=True)
st.caption("Bold = best detector on each dataset.")

# ---- attack before / after ----
st.subheader("Before and after paraphrase attacks")
long = attack.melt(id_vars="detector", value_vars=["auroc_before", "auroc_after"],
                   var_name="condition", value_name="AUROC")
long["condition"] = long["condition"].map({"auroc_before": "Before", "auroc_after": "After attack"})
chart = (
    alt.Chart(long)
    .mark_bar(clip=True)
    .encode(
        x=alt.X("detector:N", title="Detector"),
        xOffset=alt.XOffset("condition:N", sort=["Before", "After attack"]),
        y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0.5, 1])),
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
    st.caption("E will add results/findings.txt (one finding per line).")
