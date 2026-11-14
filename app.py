"""AI Text Detector — entry point (F).

Run from the repo root:  streamlit run app.py
Sets the page names shown in the sidebar.
"""
import streamlit as st

st.set_page_config(page_title="AI Text Detector", layout="wide")

nav = st.navigation([
    st.Page("detect.py", title="Detect", default=True),
    st.Page("pages/1_Results.py", title="Results"),
    st.Page("pages/2_Attack_Demo.py", title="Attack demo"),
])
nav.run()
