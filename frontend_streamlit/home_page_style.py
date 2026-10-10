import streamlit as st

def import_home_page_css_style():
    if st.session_state.selected_pipeline == "siloDesign":
        tune_border = "2px solid linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        tune_bg = "linear-gradient(45deg, transparent 5%, #1e3c72)"
        tune_color = "white"
        tune_shadow = "0 4px 15px rgba(102, 126, 234, 0.3)"
        tune_transform = "translateY(-2px)"
    else:
        tune_border = "1px solid #e0e0e0"
        tune_bg = "transparent"
        tune_color = "#555555"
        tune_shadow = "none"
        tune_transform = "translateY(0px)"

    # Design Button Styles (Column 2)
    if st.session_state.selected_pipeline == "muloDesign":
        design_border = "2px solid linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        design_bg = "linear-gradient(45deg, transparent, #f5576c)"
        design_color = "white"
        design_shadow = "0 4px 15px rgba(245, 87, 108, 0.3)"
        design_transform = "translateY(-2px)"
    else:
        design_border = "1px solid #e0e0e0"
        design_bg = "transparent"
        design_color = "#555555"
        design_shadow = "none"
        design_transform = "translateY(0px)"

    st.markdown(f"""
    <style>
        /* Target the first button column (Tune) */
        div.element-container:has(#tune-btn-marker) + div.element-container .stButton > button {{
            border: {tune_border} !important;
            background: {tune_bg} !important;
            color: {tune_color} !important;
            font-weight: bold !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: {tune_shadow} !important;
            transform: {tune_transform} !important;
            position: relative !important;
            overflow: hidden !important;
        }}

        /* Hover effect for Tune button */
        div.element-container:has(#tune-btn-marker) + div.element-container .stButton > button:hover
         {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            border-color: #667eea !important;
        }}

        /* Target the second button column (Design) */
        div.element-container:has(#design-btn-marker) + div.element-container .stButton > button {{
            border: {design_border} !important;
            background: {design_bg} !important;
            color: {design_color} !important;
            font-weight: bold !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: {design_shadow} !important;
            transform: {design_transform} !important;
            position: relative !important;
            overflow: hidden !important;
        }}

        /* Hover effect for Design button */
        div.element-container:has(#design-btn-marker) + div.element-container .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4) !important;
            border-color: #f5576c !important;
        }}

        /* Active/click effect for Design button */
        div.element-container:has(#tune-btn-marker) + div.element-container .stButton > button:active,
        div.element-container:has(#design-btn-marker) + div.element-container .stButton > button:active {{
            transform: translateY(0px) !important;
        }}

        /* Base layout and typography strictly for targeted buttons */
        div.element-container:has(#tune-btn-marker) + div.element-container .stButton > button,
        div.element-container:has(#design-btn-marker) + div.element-container .stButton > button {{
            font-size: 1rem !important;
        }}
        
        
        
        div.element-container:has(#blue_btn) + div.element-container .stButton > button {{
            border: 1px solid #e0e0e0 !important;
            background: transparent !important;
            color: #555555 !important;
            font-weight: bold !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: none !important;
            transform: translateY(0px) !important;
            position: relative !important;
            overflow: hidden !important;
        }}

        /* Hover effect for Tune button */
        div.element-container:has(#blue_btn) + div.element-container .stButton > button:hover
         {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            border-color: #667eea !important;
        }}
        
        /* Target the second button column (Design) */
        div.element-container:has(#red_btn) + div.element-container .stButton > button {{
            border: 1px solid #e0e0e0 !important;
            background: transparent !important;
            color: #555555 !important;
            font-weight: bold !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: none !important;
            transform: translateY(0px) !important;
            position: relative !important;
            overflow: hidden !important;
        }}

        /* Hover effect for Design button */
        div.element-container:has(#red_btn) + div.element-container .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4) !important;
            border-color: #f5576c !important;
        }}

        /* Active/click effect for Design button */
        div.element-container:has(#blue_btn) + div.element-container .stButton > button:active,
        div.element-container:has(#red_btn) + div.element-container .stButton > button:active {{
            transform: translateY(0px) !important;
        }}

        /* Base layout and typography strictly for targeted buttons */
        div.element-container:has(#blue_btn) + div.element-container .stButton > button,
        div.element-container:has(#red_btn) + div.element-container .stButton > button {{
            font-size: 1rem !important;
        }}
    </style>
    """, unsafe_allow_html=True)