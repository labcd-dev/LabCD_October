"""
Design tokens and CSS for the AgentSysID UI.

The palette is the one in ``frontend_mockup/labcd_sysid.html``: a quiet,
near-black engineering surface with a single indigo accent used sparingly, IBM
Plex Sans for prose and IBM Plex Mono for anything that is literally data.

The two chart series colors are validated against the dark chart surface
(#131316) on all six checks — lightness band, chroma floor, colour-vision
separation, normal-vision floor and contrast.
"""

from __future__ import annotations

# --- Surfaces ---------------------------------------------------------------
BG = "#0b0b0d"
BG_RAISED = "#101013"
BG_CARD = "#131316"
BG_INPUT = "#0f0f12"
BORDER = "#212126"
BORDER_SOFT = "#1a1a1e"

# --- Ink --------------------------------------------------------------------
TEXT = "#e9e8e4"
TEXT_DIM = "#949398"
TEXT_FAINT = "#5c5b61"

# --- Accent & status --------------------------------------------------------
ACCENT = "#6e79f0"
ACCENT_SOFT = "rgba(110,121,240,0.14)"
ACCENT_SOFT_2 = "rgba(110,121,240,0.28)"
GOOD = "#4bd1a0"
WARN = "#e0a94f"
CRIT = "#e06a5a"

# --- Chart series (validated for the dark surface) --------------------------
SERIES_A = "#6e79f0"   # train
SERIES_B = "#d4794a"   # validation

SANS = "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

#: Status -> colour, for the deployment verdict pill.
STATUS_COLORS = {
    "STABLE & HIGH-FIDELITY": GOOD,
    "STABLE & ACCEPTABLE": GOOD,
    "UNSTABLE ROLLOUT": WARN,
    "UNSTABLE / FAILED": CRIT,
}


def status_color(status: str) -> str:
    return STATUS_COLORS.get((status or "").strip().upper(), TEXT_DIM)


def score_color(score: float) -> str:
    if score > 75:
        return GOOD
    if score > 50:
        return GOOD
    if score >= 40:
        return WARN
    return CRIT


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp {{ background:{BG}; color:{TEXT}; }}
html, body, [class*="css"] {{ font-family:{SANS}; }}
code, pre, .mono {{ font-family:{MONO} !important; }}
h1, h2, h3, h4 {{ color:{TEXT}; font-weight:600; letter-spacing:-0.01em; }}

/* Streamlit chrome */
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
div.block-container {{ padding-top:1.2rem; padding-bottom:3rem; max-width:1500px; }}
#MainMenu, footer {{ visibility:hidden; }}
/* Streamlit's Deploy button sits on top of the brand bar */
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
button[data-testid="stBaseButton-headerNoPadding"] {{ display:none !important; }}
[data-testid="stAppDeployButton"] {{ display:none !important; }}

/* ---------- Brand bar ---------- */
.brandbar {{
  display:flex; align-items:center; gap:14px;
  padding:0 0 14px; margin-bottom:18px;
  border-bottom:1px solid {BORDER};
}}
.brand-mark {{
  width:34px; height:34px; border-radius:9px; flex:none;
  background:linear-gradient(135deg, {ACCENT} 0%, #8f7bf5 100%);
  display:flex; align-items:center; justify-content:center;
  color:#fff; font-weight:600; font-size:16px; letter-spacing:-0.02em;
}}
.brand-text {{ display:flex; flex-direction:column; line-height:1.25; }}
.brand-name {{
  font-size:16px; font-weight:600; color:{TEXT}; letter-spacing:0.01em;
}}
.brand-name .dim {{ color:{TEXT_FAINT}; font-weight:400; }}
.brand-sub {{
  font-family:{MONO}; font-size:11px; color:{TEXT_DIM};
  text-transform:uppercase; letter-spacing:0.13em;
}}
.brand-spacer {{ flex:1; }}
.brand-pill {{
  font-family:{MONO}; font-size:10.5px; letter-spacing:0.08em;
  padding:5px 11px; border-radius:999px;
  background:{ACCENT_SOFT}; color:{ACCENT}; border:1px solid {ACCENT_SOFT_2};
}}
.brand-status {{
  font-family:{MONO}; font-size:11px; color:{TEXT_DIM};
  display:flex; align-items:center; gap:7px;
}}
.dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.dot-idle {{ background:{TEXT_FAINT}; }}
.dot-run {{ background:{ACCENT}; animation:pulse 1.4s ease-in-out infinite; }}
.dot-done {{ background:{GOOD}; }}
.dot-fail {{ background:{CRIT}; }}
@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}

/* ---------- Sidebar (history) ---------- */
section[data-testid="stSidebar"] {{
  background:{BG_CARD}; border-right:1px solid {BORDER};
}}
section[data-testid="stSidebar"] label, .stApp label {{ color:{TEXT} !important; }}
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] {{ color:{TEXT_DIM} !important; }}
.side-heading {{
  font-family:{MONO}; font-size:10.5px; color:{TEXT_FAINT};
  text-transform:uppercase; letter-spacing:0.12em; margin:14px 0 6px;
}}
.hist-row {{ margin:14px 0 4px; }}
.hist-line {{ display:flex; align-items:baseline; gap:8px; }}
.hist-score {{ font-family:{MONO}; font-size:13px; font-weight:500; flex:none; }}
.hist-name {{
  font-size:12.5px; color:{TEXT}; flex:1;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}}
.hist-age {{ font-family:{MONO}; font-size:10px; color:{TEXT_FAINT}; flex:none; }}
.hist-meta {{
  font-family:{MONO}; font-size:10px; color:{TEXT_FAINT};
  margin:2px 0 6px 25px;
}}
.hist-empty {{
  color:{TEXT_FAINT}; font-size:12.5px; line-height:1.6;
  border:1px dashed {BORDER}; border-radius:10px; padding:14px; text-align:center;
}}

/* ---------- Cards & metrics ---------- */
div[data-testid="stMetric"] {{
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:12px; padding:15px 17px;
}}
div[data-testid="stMetricLabel"] {{
  color:{TEXT_DIM}; font-size:10.5px;
  text-transform:uppercase; letter-spacing:0.07em;
}}
div[data-testid="stMetricValue"] {{ font-family:{MONO}; letter-spacing:-0.02em; }}
div[data-testid="stExpander"] {{
  border:1px solid {BORDER}; border-radius:12px; background:{BG_CARD};
}}
div[data-testid="stExpander"] summary {{ font-size:13px; }}

.card {{
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:12px; padding:16px 18px; margin-bottom:12px;
}}
.card-title {{
  font-family:{MONO}; font-size:10.5px; color:{TEXT_FAINT};
  text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;
}}

/* ---------- Hero score ---------- */
.hero {{
  display:flex; align-items:center; gap:22px;
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:14px; padding:20px 24px; margin-bottom:14px;
}}
.hero-score {{ font-family:{MONO}; font-size:46px; font-weight:500; line-height:1; }}
.hero-of {{ font-family:{MONO}; font-size:14px; color:{TEXT_FAINT}; }}
.hero-meta {{ display:flex; flex-direction:column; gap:7px; }}
.hero-sub {{ font-family:{MONO}; font-size:11.5px; color:{TEXT_DIM}; }}

/* ---------- Pills ---------- */
.pill {{
  display:inline-block; padding:4px 11px; border-radius:999px;
  font-family:{MONO}; font-size:10.5px; letter-spacing:0.06em;
}}
.tag {{
  display:inline-block; padding:2px 9px; border-radius:6px; margin-right:6px;
  font-family:{MONO}; font-size:10.5px; color:{TEXT_DIM};
  background:{BG_INPUT}; border:1px solid {BORDER};
}}

/* ---------- Stage rail ---------- */
.rail {{ display:flex; flex-wrap:wrap; gap:6px 0; margin:4px 0 10px; }}
.rail-step {{
  font-family:{MONO}; font-size:11px; display:flex; align-items:center; gap:6px;
  padding-right:14px;
}}
.rail-done {{ color:{GOOD}; }}
.rail-active {{ color:{ACCENT}; font-weight:500; }}
.rail-todo {{ color:{TEXT_FAINT}; }}

/* ---------- Nav (segmented control) ---------- */
div[data-testid="stSegmentedControl"] button {{
  font-family:{MONO}; font-size:12px; letter-spacing:0.04em;
}}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {{
  font-family:{MONO}; font-size:12.5px;
}}
.stButton>button {{ border-radius:8px; border:1px solid {BORDER}; font-weight:500; }}
.stDownloadButton>button {{
  border-radius:8px; border:1px solid {BORDER}; background:{BG_CARD};
  font-family:{MONO}; font-size:12px;
}}
.stDownloadButton>button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
hr {{ border-color:{BORDER}; }}
</style>
"""


def brand_bar(subtitle: str, status_label: str, status_kind: str = "idle") -> str:
    """The LabCD header: wordmark, product line, module badge and run status."""
    return f"""
    <div class="brandbar">
      <div class="brand-mark">L</div>
      <div class="brand-text">
        <div class="brand-name">LabCD<span class="dim">.ai</span></div>
        <div class="brand-sub">{subtitle}</div>
      </div>
      <div class="brand-spacer"></div>
      <span class="brand-pill">AgentSysID</span>
      <span class="brand-status"><span class="dot dot-{status_kind}"></span>{status_label}</span>
    </div>
    """
