import os
import streamlit as st

st.set_page_config(page_title="Reporting Factory – Rule Authoring", layout="wide")

# --- Show homepage banner image ---
st.image("homepage_banner.png", use_column_width=True)

# ---- Sidebar title customization ----
st.markdown("""
    <style>
        /* Change default 'App' label in sidebar nav */
        [data-testid="stSidebarNav"]::before {
            content: "🧩 Rule Editor";
            margin-left: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)