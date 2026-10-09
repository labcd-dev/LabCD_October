#!/usr/bin/env python3
"""LabCD conversation workspace. Launch with run_agent_sysid_ui.py."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frontend_streamlit.ui_conversation import render_app
from frontend_streamlit.ui_conversation_theme import CSS
from frontend_streamlit import ui_brand
from frontend_streamlit.ui_pipeline_runtime import PipelineRunner, drain

st.set_page_config(page_title="LabCD · Conversation workspace", page_icon=str(ui_brand.ICON_PATH),
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
render_app(runner=PipelineRunner, drain=drain)
