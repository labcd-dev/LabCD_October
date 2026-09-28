"""
Design tokens and CSS for the AgentSysID UI.

The UI uses a near-black IDE surface, terminal teal accents, IBM Plex Sans for
prose, and IBM Plex Mono for labels and data.
"""

from __future__ import annotations

# --- Surfaces ---------------------------------------------------------------
BG = "#080a0d"
BG_RAISED = "#0e1115"
BG_CARD = "#12161b"
BG_INPUT = "#0b0e12"
BORDER = "#262c33"
BORDER_SOFT = "#1b2026"

# --- Ink --------------------------------------------------------------------
TEXT = "#d4d4d4"
TEXT_DIM = "#9da5ad"
TEXT_FAINT = "#6a737d"

# --- Accent & status --------------------------------------------------------
ACCENT = "#4ec9b0"
ACCENT_SOFT = "rgba(78,201,176,0.14)"
ACCENT_SOFT_2 = "rgba(78,201,176,0.28)"
GOOD = "#73c991"
WARN = "#dcdcaa"
CRIT = "#f48771"

# --- Chart series (validated for the dark surface) --------------------------
SERIES_A = "#4ec9b0"   # train
SERIES_B = "#ce9178"   # validation

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
h1, h2, h3, h4 {{
  color:{TEXT}; font-family:{MONO}; font-weight:500; letter-spacing:0;
}}

/* Streamlit chrome */
header[data-testid="stHeader"] {{
  background:transparent; height:0; overflow:visible; pointer-events:none;
}}
div.block-container {{ padding-top:1.2rem; padding-bottom:3rem; max-width:1500px; }}
#MainMenu, footer {{ visibility:hidden; }}
/* Keep the native sidebar controls available while hiding the other chrome. */
div[data-testid="stDecoration"], [data-testid="stToolbarActions"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {{
  display:none !important;
}}
div[data-testid="stToolbar"] {{ height:0; overflow:visible; }}

/* Panel icon on Streamlit's own buttons: toggling stays entirely client-side. */
[data-testid="stSidebarCollapseButton"] {{ visibility:visible !important; }}
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="stExpandSidebarButton"] {{
  display:inline-flex !important; align-items:center; justify-content:center;
  width:34px; height:34px; padding:0; border:1px solid {BORDER};
  border-radius:6px; background:{BG_CARD}; color:{TEXT_DIM};
  pointer-events:auto; cursor:pointer;
  transition:background 150ms ease, border-color 150ms ease, color 150ms ease;
}}
button[data-testid="stExpandSidebarButton"] {{
  position:fixed; top:12px; left:16px; z-index:1001;
}}
/* Leave room above the brand mark for the reopen control on narrow screens. */
.stApp:has(button[data-testid="stExpandSidebarButton"]) div.block-container {{
  padding-top:2.2rem;
}}
[data-testid="stSidebarCollapseButton"] button:hover,
button[data-testid="stExpandSidebarButton"]:hover {{
  background:{ACCENT_SOFT}; border-color:{ACCENT}; color:{ACCENT};
}}
[data-testid="stSidebarCollapseButton"] button:focus-visible,
button[data-testid="stExpandSidebarButton"]:focus-visible {{
  outline:2px solid {ACCENT}; outline-offset:3px;
}}
[data-testid="stSidebarCollapseButton"] button > *,
button[data-testid="stExpandSidebarButton"] > * {{ display:none; }}
[data-testid="stSidebarCollapseButton"] button::before,
button[data-testid="stExpandSidebarButton"]::before {{
  content:""; width:19px; height:19px; background:currentColor;
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2'/%3E%3Cpath d='M9 4v16'/%3E%3C/svg%3E") center / contain no-repeat;
}}
/* Visually hidden names also make the icon buttons understandable to readers. */
[data-testid="stSidebarCollapseButton"] button::after,
button[data-testid="stExpandSidebarButton"]::after {{
  position:absolute; width:1px; height:1px; overflow:hidden;
  clip-path:inset(50%); white-space:nowrap;
}}
[data-testid="stSidebarCollapseButton"] button::after {{ content:"Hide run history"; }}
button[data-testid="stExpandSidebarButton"]::after {{ content:"Show run history"; }}

/* ---------- Brand bar ---------- */
.brandbar {{
  display:flex; align-items:center; gap:14px;
  padding:0 0 14px; margin-bottom:18px;
  border-bottom:1px solid {BORDER};
}}
.brand-mark {{
  width:34px; height:34px; border-radius:6px; flex:none;
  background:linear-gradient(135deg, {ACCENT} 0%, #2d8c7a 100%);
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
.st-key-history_list [data-testid="stCaptionContainer"] p {{
  font-family:{MONO}; font-size:9.5px; color:{TEXT_FAINT};
  text-transform:uppercase; letter-spacing:0.1em; padding:10px 8px 0;
}}
/* Scope row styling to public widget/container keys. */
[class*="st-key-history-row-"] {{
  border-radius:7px; transition:background 140ms ease; overflow:hidden;
}}
[class*="st-key-history-row-"]:hover {{ background:rgba(255,255,255,0.035); }}
[class*="st-key-history-row-selected-"] {{
  background:{ACCENT_SOFT}; box-shadow:inset 3px 0 {ACCENT};
}}
[class*="st-key-history-row-selected-"]:hover {{ background:{ACCENT_SOFT_2}; }}
[class*="st-key-history-row-"] [data-testid="stColumn"] {{ min-width:0 !important; }}
[class*="st-key-history-row-"] [data-testid="stColumn"]:last-child {{
  flex:0 0 34px; width:34px;
}}
[class*="st-key-history-row-"] .stButton button {{
  width:100%; height:58px; border:0; border-radius:0; padding:8px 10px;
  background:transparent; color:{TEXT}; justify-content:flex-start;
  text-align:left; box-shadow:none;
}}
[class*="st-key-history-row-"] .stButton button:focus-visible {{
  outline:2px solid {ACCENT}; outline-offset:-2px;
}}
[class*="st-key-history-row-"] .stButton [data-testid="stMarkdownContainer"] {{
  width:100%; overflow:hidden; text-align:left;
}}
[class*="st-key-history-row-"] .stButton [data-testid="stMarkdownContainer"] p {{
  margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-family:{SANS}; font-size:12.5px; line-height:1.55;
}}
[class*="st-key-history-row-"] .stButton [data-testid="stMarkdownContainer"] p + p {{
  font-family:{MONO}; font-size:9.5px; color:{TEXT_FAINT}; font-weight:400;
}}
[class*="st-key-history-row-"] .stButton [data-testid="stIconMaterial"] {{
  color:{ACCENT}; font-size:16px;
}}
[class*="st-key-history-menu-"] [data-testid="stPopoverButton"] {{
  padding:0; min-height:34px; height:34px; border:0; background:transparent;
  color:{TEXT_DIM}; opacity:0.6;
}}
[class*="st-key-history-row-"]:hover [data-testid="stPopoverButton"],
[class*="st-key-history-row-"]:focus-within [data-testid="stPopoverButton"] {{ opacity:1; }}
[class*="st-key-history-menu-"] [data-testid="stPopoverButton"]:hover {{
  color:{ACCENT}; background:{ACCENT_SOFT};
}}
/* Hide the visible menu label and chevron; retain its accessible action name. */
[class*="st-key-history-menu-"] [data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"] {{
  position:absolute; width:1px; height:1px; overflow:hidden;
  clip-path:inset(50%); white-space:nowrap;
}}
[class*="st-key-history-menu-"] [data-testid="stPopoverButton"] [aria-hidden="true"] {{ display:none; }}
.hist-empty {{
  color:{TEXT_FAINT}; font-size:12.5px; line-height:1.6;
  border:1px dashed {BORDER}; border-radius:10px; padding:14px; text-align:center;
}}

/* ---------- Cards & metrics ---------- */
div[data-testid="stMetric"] {{
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:7px; padding:15px 17px;
}}
div[data-testid="stMetricLabel"] {{
  color:{TEXT_DIM}; font-size:10.5px;
  text-transform:uppercase; letter-spacing:0.07em;
}}
div[data-testid="stMetricValue"] {{ font-family:{MONO}; letter-spacing:-0.02em; }}
div[data-testid="stExpander"] {{
  border:1px solid {BORDER}; border-radius:7px; background:{BG_CARD};
}}
div[data-testid="stExpander"] summary {{
  font-family:{MONO}; font-size:12px; color:{TEXT_DIM};
}}

.card {{
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:7px; padding:16px 18px; margin-bottom:12px;
}}
.card-title {{
  font-family:{MONO}; font-size:10.5px; color:{TEXT_FAINT};
  text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;
}}

/* ---------- Hero score ---------- */
.hero {{
  display:flex; align-items:center; gap:22px;
  background:{BG_CARD}; border:1px solid {BORDER};
  border-radius:8px; padding:20px 24px; margin-bottom:14px;
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

/* ---------- Configure overview: animated system-identification schematic ---------- */
.sysid-intro {{
  display:grid; grid-template-columns:minmax(190px,0.76fr) minmax(0,1.34fr);
  gap:16px; align-items:center; overflow:hidden;
  background:linear-gradient(118deg,#0d1418 0%,#10191e 54%,#0b1115 100%);
  border:1px solid {BORDER}; border-radius:10px; padding:18px 20px;
  margin:4px 0 20px; box-shadow:0 14px 38px rgba(0,0,0,0.18);
}}
.sysid-intro-copy {{ min-width:0; padding:4px 2px; }}
.sysid-intro-kicker, .sysid-intro-footnote {{
  color:{ACCENT}; font-family:{MONO}; font-size:9px; font-weight:500;
  letter-spacing:0.11em; line-height:1.6;
}}
.sysid-intro-title {{
  color:{TEXT}; font-family:{SANS} !important; font-size:clamp(21px,2vw,29px);
  font-weight:600; letter-spacing:-0.035em; line-height:1.14;
  margin:10px 0 10px;
}}
.sysid-intro-definition {{
  color:{TEXT_DIM}; font-size:12.5px; line-height:1.65; max-width:460px;
  margin:0 0 14px;
}}
.sysid-intro-definition strong {{ color:{TEXT}; font-weight:600; }}
.sysid-intro-features {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }}
.sysid-intro-features span {{
  border:1px solid {BORDER}; border-radius:5px; background:{BG_INPUT};
  color:{TEXT_DIM}; font:9px {MONO}; padding:5px 7px; white-space:nowrap;
}}
.sysid-visual {{
  min-width:0; overflow:hidden; border:1px solid {BORDER}; border-radius:8px;
  background:#0b1115; padding:12px 13px 10px;
}}
.scene-header, .scene-node-head, .scene-legend {{
  display:flex; align-items:center; justify-content:space-between; gap:8px;
}}
.scene-header {{ margin:0 1px 11px; }}
.scene-label, .scene-state {{
  color:{TEXT_FAINT}; font:8px {MONO}; letter-spacing:0.1em;
}}
.scene-label {{ color:{ACCENT}; font-weight:600; }}
.scene-live-dot {{
  display:inline-block; width:6px; height:6px; margin-right:5px; border-radius:50%;
  background:{ACCENT}; box-shadow:0 0 9px {ACCENT}; animation:scene-pulse 1.8s ease-in-out infinite;
}}
.scene-flow-row {{
  display:grid; grid-template-columns:minmax(0,1fr) 42px minmax(0,1fr) 42px minmax(0,1fr);
  align-items:stretch; gap:0;
}}
.scene-node {{
  position:relative; min-width:0; padding:11px 10px 9px; overflow:hidden;
  border:1px solid #303b42; border-radius:6px;
  background:linear-gradient(145deg,#141b20 0%,#101519 100%);
  box-shadow:inset 0 1px rgba(255,255,255,0.025);
}}
.scene-node::before, .scene-node::after {{
  content:""; position:absolute; top:50%; width:6px; height:6px; margin-top:-3px;
  border:1px solid #71818a; border-radius:50%; background:#0b1115;
}}
.scene-node::before {{ left:-4px; }}
.scene-node::after {{ right:-4px; }}
.data-block {{ border-color:rgba(78,201,176,0.48); }}
.model-block {{ border-color:#4a5861; }}
.verify-block {{ border-color:rgba(86,156,214,0.48); }}
.scene-node-head {{
  color:{TEXT_FAINT}; font:7px {MONO}; letter-spacing:0.08em; white-space:nowrap;
  margin-bottom:12px;
}}
.scene-node-head b {{
  color:{ACCENT}; font:7px {MONO}; letter-spacing:0.06em;
  border:1px solid {ACCENT_SOFT_2}; border-radius:3px; padding:3px 4px;
}}
.verify-block .scene-node-head b {{ color:#7eb9e8; border-color:rgba(86,156,214,0.35); }}
.scene-equation {{ color:{TEXT}; font:600 14px {MONO}; letter-spacing:-0.03em; white-space:nowrap; }}
.model-block .scene-equation {{ color:{ACCENT}; font-size:15px; }}
.scene-copy {{ color:{TEXT_DIM}; font:9px {SANS}; line-height:1.45; margin-top:4px; min-height:25px; }}
.scene-wave {{
  display:flex; align-items:center; gap:3px; height:35px; margin:8px 0 7px; padding:0 2px;
  border-bottom:1px solid {BORDER};
  background:repeating-linear-gradient(to bottom,transparent 0,transparent 11px,rgba(86,101,110,0.14) 12px);
}}
.scene-wave i {{
  display:block; flex:1; min-width:2px; border-radius:2px 2px 0 0;
  background:linear-gradient(180deg,#65d8c0,{ACCENT}); opacity:0.82;
  animation:sample-glow 2.4s ease-in-out infinite alternate;
}}
.scene-wave i:nth-child(3n) {{ animation-delay:0.35s; }}
.scene-wave i:nth-child(4n) {{ animation-delay:0.7s; }}
.scene-node-foot {{ color:{TEXT_FAINT}; font:7px {MONO}; letter-spacing:0.035em; white-space:nowrap; }}
.scene-params {{ display:flex; gap:6px; margin:11px 0 12px; }}
.scene-params span {{
  display:grid; place-items:center; width:31px; height:27px; border:1px solid #35464c;
  border-radius:4px; color:{TEXT_DIM}; background:{BG_INPUT}; font:11px {MONO};
}}
.scene-params .param-focus {{ color:{ACCENT}; border-color:{ACCENT}; background:{ACCENT_SOFT}; }}
.compare-traces {{ display:grid; gap:5px; margin:10px 0 7px; }}
.compare-row {{ display:flex; align-items:center; gap:5px; }}
.compare-label {{ flex:none; color:{TEXT_FAINT}; font:6px {MONO}; width:28px; }}
.compare-bars {{ display:flex; align-items:center; gap:3px; height:12px; flex:1; }}
.compare-bars i {{ display:block; flex:1; min-width:2px; border-radius:2px; background:#569cd6; opacity:0.9; }}
.compare-model i {{ background:#c586c0; opacity:0.78; }}
.scene-wire {{
  position:relative; min-width:0; display:flex; flex-direction:column;
  justify-content:center; align-items:center; gap:6px;
}}
.wire-label, .wire-type {{ color:{TEXT_FAINT}; font:7px {MONO}; letter-spacing:0.05em; }}
.wire-track {{ position:relative; width:100%; height:1px; background:#344149; }}
.wire-track::after {{
  content:""; position:absolute; right:0; top:-3px; border-top:3px solid transparent;
  border-bottom:3px solid transparent; border-left:4px solid {ACCENT};
}}
.wire-track i {{
  position:absolute; z-index:1; top:-3px; left:0; width:7px; height:7px;
  border-radius:50%; background:{ACCENT}; box-shadow:0 0 7px {ACCENT};
  animation:signal-travel 1.7s ease-in-out infinite;
}}
.scene-wire-test .wire-track i {{ animation-delay:0.55s; background:#7eb9e8; box-shadow:0 0 7px #569cd6; }}
.scene-wire-test .wire-track::after {{ border-left-color:#569cd6; }}
.scene-legend {{ justify-content:flex-start; flex-wrap:wrap; gap:12px; margin:10px 2px 0; }}
.scene-legend span {{ color:{TEXT_FAINT}; font:7px {MONO}; }}
.scene-legend span i {{
  display:inline-block; width:6px; height:6px; margin-right:5px; border-radius:50%;
}}
.legend-measured {{ background:{ACCENT}; }}
.legend-model {{ background:#c586c0; }}
.scene-legend .scene-loop {{ margin-left:auto; color:{TEXT_DIM}; letter-spacing:0.08em; }}
@keyframes signal-travel {{
  0% {{ left:0; opacity:0.35; }}
  45% {{ opacity:1; }}
  100% {{ left:calc(100% - 7px); opacity:0.35; }}
}}
@keyframes scene-pulse {{ 0%,100% {{ opacity:0.55; }} 50% {{ opacity:1; }} }}
@keyframes sample-glow {{ from {{ opacity:0.55; }} to {{ opacity:0.95; }} }}
@media (max-width:760px) {{
  .sysid-intro {{ grid-template-columns:1fr; gap:12px; }}
  .sysid-intro-copy {{ padding:0; }}
  .sysid-intro-definition {{ max-width:760px; }}
}}
@media (max-width:1200px) {{
  .scene-flow-row {{ grid-template-columns:1fr; gap:0; }}
  .scene-wire {{ height:27px; }}
  .wire-label {{ position:absolute; left:calc(50% + 10px); top:1px; }}
  .wire-type {{ position:absolute; right:calc(50% + 10px); top:1px; }}
  .wire-track {{ width:1px; height:22px; }}
  .wire-track::after {{
    right:-2px; top:auto; bottom:0; border-left:3px solid transparent;
    border-right:3px solid transparent; border-top:4px solid {ACCENT}; border-bottom:0;
  }}
  .wire-track i {{ top:0; left:-3px; animation:signal-travel-vertical 1.7s ease-in-out infinite; }}
  .scene-wire-test .wire-track::after {{ border-top-color:#569cd6; border-left-color:transparent; }}
  @keyframes signal-travel-vertical {{
    0% {{ top:0; opacity:0.35; }} 45% {{ opacity:1; }} 100% {{ top:15px; opacity:0.35; }}
  }}
}}
@media (prefers-reduced-motion:reduce) {{
  .scene-live-dot, .scene-wave i, .wire-track i {{ animation:none; }}
  .wire-track i {{ left:calc(100% - 7px); }}
}}

/* ---------- Client-facing MLP / LSTM architecture sketches ---------- */
.model-diagrams {{
  display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px;
  margin:3px 0 11px;
}}
.model-diagram {{
  min-width:0; border:1px solid {BORDER}; border-radius:7px; padding:9px 9px 7px;
  background:linear-gradient(145deg,#10161b,#0b1014);
  transition:border-color 160ms ease, background-color 160ms ease;
}}
.model-diagram.selected {{
  border-color:rgba(78,201,176,0.75);
  background:linear-gradient(145deg,rgba(78,201,176,0.09),#0b1014 72%);
  box-shadow:inset 0 0 0 1px rgba(78,201,176,0.08);
}}
.model-diagram-head {{
  display:flex; align-items:center; justify-content:space-between; gap:5px;
  margin-bottom:9px; color:{TEXT_FAINT}; font:9px {MONO};
}}
.model-diagram-head b {{ color:{TEXT}; font:600 11px {MONO}; letter-spacing:0.04em; }}
.model-diagram.selected .model-diagram-head b {{ color:{ACCENT}; }}
.mlp-flow {{
  min-height:66px; display:grid; grid-template-columns:minmax(42px,1fr) 12px 24px 12px minmax(53px,1fr);
  align-items:center; justify-items:center; gap:3px;
}}
.mlp-input {{ width:100%; display:grid; gap:4px; }}
.mlp-input i, .mlp-output {{
  display:grid; place-items:center; min-height:23px; padding:2px 4px;
  border:1px solid #33434b; border-radius:4px; color:{TEXT_DIM};
  background:#11191e; font:8px {MONO}; font-style:normal; text-align:center;
}}
.mlp-output {{ border-color:rgba(78,201,176,0.38); color:{ACCENT}; }}
.model-arrow {{ color:{ACCENT}; font:12px {MONO}; text-align:center; }}
.mlp-layers {{ display:flex; flex-direction:column; align-items:center; gap:3px; }}
.mlp-layers i {{
  display:block; width:17px; height:17px; border:1px solid #569cd6; border-radius:50%;
  background:radial-gradient(circle at 35% 35%,rgba(126,185,232,0.32),rgba(86,156,214,0.08));
}}
.lstm-flow {{ min-height:66px; display:flex; flex-direction:column; justify-content:center; gap:6px; }}
.lstm-steps {{ display:flex; align-items:center; justify-content:center; gap:5px; }}
.lstm-steps i {{
  display:grid; place-items:center; width:36px; height:25px; border:1px solid #3c5256;
  border-radius:4px; color:{ACCENT}; background:#11191e; font:9px {MONO}; font-style:normal;
}}
.lstm-steps span {{ color:{TEXT_FAINT}; font:9px {MONO}; }}
.lstm-memory {{ display:flex; align-items:center; gap:6px; color:{TEXT_FAINT}; font:8px {MONO}; }}
.lstm-memory i {{
  position:relative; display:block; flex:1; height:2px; border-radius:2px;
  background:linear-gradient(90deg,rgba(78,201,176,0.22),{ACCENT});
}}
.lstm-memory i::after {{
  content:""; position:absolute; right:0; top:-2px; width:6px; height:6px;
  border-radius:50%; background:{ACCENT}; box-shadow:0 0 7px rgba(78,201,176,0.45);
}}
.lstm-result {{ color:{TEXT_DIM}; font:8px {MONO}; text-align:right; }}
.model-diagram p {{ margin:7px 0 0; color:{TEXT_FAINT}; font:9px {SANS}; line-height:1.4; }}
@media (max-width:1050px) {{ .model-diagrams {{ grid-template-columns:1fr; }} }}

/* ---------- Monitor: Simulink-inspired pipeline blocks ---------- */
.pipeline-board {{
  position:relative; overflow:hidden; margin:2px 0 16px; padding:16px 17px 14px;
  border:1px solid {BORDER}; border-radius:9px; background-color:{BG_CARD};
  background-image:radial-gradient(rgba(115,130,140,0.16) 0.8px, transparent 0.8px);
  background-size:18px 18px;
}}
.pipeline-board-head {{
  display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
  padding:0 1px 3px;
}}
.pipeline-kicker {{
  color:{ACCENT}; font:9px {MONO}; font-weight:600; letter-spacing:0.13em;
  text-transform:uppercase; margin-bottom:5px;
}}
.pipeline-heading {{ color:{TEXT}; font:500 16px {MONO}; letter-spacing:-0.02em; }}
.pipeline-current {{ text-align:right; flex:none; }}
.pipeline-current-label {{
  color:{TEXT_FAINT}; font:8px {MONO}; letter-spacing:0.1em; text-transform:uppercase;
}}
.pipeline-current-value {{ color:{ACCENT}; font:11px {MONO}; margin-top:4px; }}
.pipeline-flow {{
  display:grid; grid-template-columns:minmax(0,1fr) 48px minmax(0,1fr) 48px minmax(0,1fr) 48px minmax(0,1fr);
  align-items:stretch; gap:0; margin:14px 0 15px;
}}
.pipeline-block {{
  position:relative; min-width:0; min-height:172px; overflow:visible;
  padding:12px 13px 11px; border:1px solid #303a41; border-radius:6px;
  background:linear-gradient(145deg,#141a1f 0%,#101519 100%);
  box-shadow:inset 0 1px rgba(255,255,255,0.025);
}}
.pipeline-block::before, .pipeline-block::after {{
  content:""; position:absolute; z-index:2; top:50%; width:7px; height:7px;
  margin-top:-4px; border:1px solid #56646d; border-radius:50%; background:{BG_CARD};
}}
.pipeline-block::before {{ left:-4px; }}
.pipeline-block::after {{ right:-4px; }}
.pipeline-block.done {{ border-color:rgba(115,201,145,0.36); }}
.pipeline-block.done::before, .pipeline-block.done::after {{ border-color:{GOOD}; }}
.pipeline-block.active {{
  border-color:{ACCENT}; box-shadow:inset 3px 0 {ACCENT},0 0 24px {ACCENT_SOFT};
}}
.pipeline-block.active::before, .pipeline-block.active::after {{
  border-color:{ACCENT}; background:{ACCENT}; box-shadow:0 0 9px {ACCENT};
}}
.pipe-block-meta {{ display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:11px; }}
.pipe-index {{ color:{TEXT_FAINT}; font:9px {MONO}; letter-spacing:0.08em; }}
.pipe-state {{
  color:{TEXT_FAINT}; font:8px {MONO}; letter-spacing:0.07em;
  border:1px solid {BORDER}; border-radius:3px; padding:3px 5px;
}}
.pipeline-block.done .pipe-state {{ color:{GOOD}; border-color:rgba(115,201,145,0.32); }}
.pipeline-block.active .pipe-state {{ color:{ACCENT}; border-color:{ACCENT_SOFT_2}; background:{ACCENT_SOFT}; }}
.pipe-title {{ color:{TEXT}; font:600 12px {MONO}; line-height:1.4; margin-bottom:4px; }}
.pipe-subtitle {{ color:{ACCENT}; font:8px {MONO}; line-height:1.5; min-height:25px; }}
.pipe-description {{ color:{TEXT_DIM}; font:10px {SANS}; line-height:1.5; margin:6px 0 10px; }}
.pipe-tags {{ display:flex; flex-wrap:wrap; gap:4px; }}
.pipe-tags span {{
  color:{TEXT_FAINT}; background:{BG_INPUT}; border:1px solid {BORDER_SOFT};
  border-radius:3px; padding:3px 5px; font:7px {MONO}; letter-spacing:0.04em;
}}
.pipeline-link {{
  position:relative; display:flex; flex-direction:column; justify-content:center;
  align-items:center; gap:5px; min-width:0; color:{TEXT_FAINT};
}}
.pipeline-link::before {{
  content:""; position:absolute; left:1px; right:5px; top:45%;
  height:1px; background:linear-gradient(90deg,{BORDER},rgba(78,201,176,0.62));
}}
.pipeline-link-icon {{
  position:relative; z-index:1; padding:0 4px; color:{ACCENT};
  background:{BG_CARD}; font:17px {MONO}; line-height:1;
}}
.pipeline-link-label {{ color:{TEXT_FAINT}; font:7px {MONO}; text-align:center; line-height:1.3; }}
.pipeline-board-footer {{ display:flex; align-items:center; gap:12px; }}
.pipeline-progress-label {{ flex:none; color:{TEXT_DIM}; font:9px {MONO}; }}
.pipeline-progress-track {{
  flex:1; height:4px; overflow:hidden; border-radius:3px; background:#252d33;
}}
.pipeline-progress-fill {{
  display:block; height:100%; border-radius:3px;
  background:linear-gradient(90deg,#2d8c7a,{ACCENT});
  box-shadow:0 0 10px {ACCENT_SOFT_2}; transition:width 450ms ease;
}}
@media (max-width:1100px) {{
  .pipeline-flow {{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
  .pipeline-link {{ display:none; }}
}}
@media (max-width:620px) {{
  .pipeline-board {{ padding:13px 11px; }}
  .pipeline-board-head {{ align-items:flex-start; flex-direction:column; }}
  .pipeline-current {{ text-align:left; }}
  .pipeline-flow {{ grid-template-columns:1fr; }}
  .pipeline-block {{ min-height:0; }}
}}

/* ---------- Nav (segmented control) ---------- */
div[data-testid="stSegmentedControl"] button {{
  font-family:{MONO}; font-size:12px; letter-spacing:0.04em;
}}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {{
  font-family:{MONO}; font-size:12.5px;
}}
.stButton>button {{
  border-radius:6px; border:1px solid {BORDER};
  font-family:{MONO}; font-size:12px; font-weight:500;
}}
.stDownloadButton>button {{
  border-radius:6px; border:1px solid {BORDER}; background:{BG_CARD};
  font-family:{MONO}; font-size:12px;
}}
.stDownloadButton>button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
pre {{ background:{BG_INPUT}; border:1px solid {BORDER}; border-radius:6px; }}
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
