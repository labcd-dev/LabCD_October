"""In-chat measurement plots, setup choices and per-state run evidence."""
from __future__ import annotations

import hashlib
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from . import conversation_analysis as analysis, ui_activity as activity, ui_history as hist
except ImportError:
    import conversation_analysis as analysis
    import ui_activity as activity
    import ui_history as hist


@st.cache_data(show_spinner=False, max_entries=24)
def _preview(path: str, digest: str, name: str) -> pd.DataFrame:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("The saved dataset changed. Attach it again to see its plots.")
    frame = analysis.read_dataset(name, data)
    if len(frame) > 800:
        frame = frame.iloc[::max(1, len(frame) // 800)].copy()
    return frame


def _signal(dataset: dict, name: str, key: str, *, height=205):
    try:
        frame = _preview(dataset["path"], dataset["sha256"], dataset["name"])
        if name not in frame:
            return
        if "time" in frame:
            st.line_chart(frame[["time", name]], x="time", y=name, x_label="Time (s)",
                          y_label=name, height=height)
        else:
            st.line_chart(frame[[name]], y=name, x_label="Sample", y_label=name, height=height)
    except (OSError, ValueError) as exc:
        st.warning(str(exc))


def _render_detection_table(title: str, summary: str, badge: str,
                            columns: list[str], rows: list[list[str]], note: str):
    if not rows:
        return
    header = "".join(f"<th scope=\"col\">{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'''<section class="detection-table-card" aria-label="{html.escape(title, quote=True)}">
  <div class="detection-table-heading">
    <div><div class="detection-table-kicker">DATA SCAN · REVIEW</div>
      <div class="detection-table-title">{html.escape(title)}</div>
      <div class="detection-table-summary">{html.escape(summary)}</div></div>
    <span class="detection-table-count">{html.escape(badge)}</span>
  </div>
  <div class="detection-table-scroll"><table class="detection-table">
    <thead><tr>{header}</tr></thead><tbody>{body}</tbody>
  </table></div>
  <div class="detection-table-note">{html.escape(note)}</div>
</section>''',
        unsafe_allow_html=True,
    )


_PICTURE_CHOICE = st.components.v2.component(
    "labcd_setup_picture_choice",
    html='<div class="setup-picture-choice" role="group"></div>',
    css="""
.setup-picture-choice { color:#dce9e5; font-family:Arial,sans-serif; }
.picture-choice-heading { margin:0 0 9px; color:#e7f1ed; font-size:14px; font-weight:650; }
.picture-choice-hint { margin:-5px 0 12px; color:#9eafaa; font-size:11px; line-height:1.5; }
.picture-choice-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.picture-choice-card {
  display:flex; width:100%; min-width:0; flex-direction:column; align-items:stretch; gap:7px;
  padding:11px; color:#e6f0ec; text-align:left; cursor:pointer;
  border:1px solid #354744; border-radius:13px;
  background:linear-gradient(145deg,#141e1d 0%,#191c22 72%,#1d1d24 100%);
  box-shadow:inset 0 1px #ffffff08;
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease,background .18s ease;
}
.picture-choice-card:hover { transform:scale(1.012); border-color:#5c877b; background:#1b2524; }
.setup-picture-choice--readonly .picture-choice-card { cursor:default; }
.setup-picture-choice--readonly .picture-choice-card:hover { transform:none; border-color:#354744; background:linear-gradient(145deg,#141e1d 0%,#191c22 72%,#1d1d24 100%); }
.setup-picture-choice--readonly .picture-choice-card[aria-pressed="true"]:hover { border-color:#66d5bd; background:linear-gradient(145deg,#1a3530 0%,#1b2828 68%,#22232b 100%); }
.setup-picture-choice--readonly .picture-choice-card:disabled { opacity:1; }
.picture-choice-card:focus-visible { outline:2px solid #7de0cb; outline-offset:3px; }
.picture-choice-card[aria-pressed="true"] {
  transform:scale(1.022); border-color:#66d5bd;
  background:linear-gradient(145deg,#1a3530 0%,#1b2828 68%,#22232b 100%);
  box-shadow:0 0 0 1px #66d5bd44,0 0 23px #49d0b42b,inset 0 1px #d8fff31b;
  animation:picture-choice-glow 2.4s ease-in-out infinite;
}
.picture-choice-art { display:block; width:100%; overflow:hidden; border:1px solid #2e403e; border-radius:9px; background:#111819; }
.picture-choice-card[aria-pressed="true"] .picture-choice-art { border-color:#598d7e; }
.picture-choice-art svg { display:block; width:100%; height:auto; }
.architecture-diagram { display:block; width:100%; height:auto; }
.architecture-diagram.unselected { opacity:.9; }
.arch-label { fill:#8db0aa; font:600 9px ui-monospace,monospace; letter-spacing:.7px; }
.arch-main { fill:#e0eeea; font:600 12px ui-monospace,monospace; }
.arch-small { fill:#9bafaa; font:10px Arial,sans-serif; }
.arch-foot { fill:#80a29b; font:10px Arial,sans-serif; }
.arch-sample { fill:#172322; stroke:#3a5a55; stroke-width:1.2; }
.arch-sample.current { fill:#1a302d; stroke:#5bbba8; }
.arch-block { fill:#18302e; stroke:#4f978b; stroke-width:1.4; }
.arch-output { fill:#202534; stroke:#707fc5; stroke-width:1.2; }
.arch-wire { fill:none; stroke:#75b7a9; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.arch-wire.faint { stroke:#4b7069; stroke-width:1.1; }
.arch-arrow { fill:none; stroke:#75b7a9; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.arch-node { fill:#263e3b; stroke:#72b7a8; stroke-width:1.4; }
.arch-node.bright { fill:#67d5bd; stroke:#b2f7e8; animation:arch-neuron-pulse 2.1s ease-in-out infinite; }
.arch-memory { fill:none; stroke:#9fe7d8; stroke-width:2; stroke-linecap:round; stroke-dasharray:5 4; animation:arch-signal-flow 2.4s linear infinite; }
.single-run-flow { stroke-dasharray:9 21; animation:picture-flow 2.2s linear infinite; }
.picture-choice-title { display:flex; align-items:center; gap:7px; font-size:13px; font-weight:700; }
.picture-choice-title::before {
  content:""; width:9px; height:9px; flex:none; border:1px solid #607873; border-radius:50%;
  background:#1c2827; transition:all .16s ease;
}
.picture-choice-card[aria-pressed="true"] .picture-choice-title::before {
  border-color:#9bf1de; background:#6de0c5; box-shadow:0 0 9px #6de0c588;
}
.picture-choice-description { color:#9eafaa; font-size:11px; line-height:1.45; }
@keyframes picture-choice-glow {
  0%,100% { box-shadow:0 0 0 1px #66d5bd33,0 0 17px #49d0b41b,inset 0 1px #d8fff310; }
  50% { box-shadow:0 0 0 1px #7de8d177,0 0 29px #49d0b444,inset 0 1px #d8fff322; }
}
@keyframes picture-flow { to { stroke-dashoffset:-60; } }
@keyframes arch-neuron-pulse { 0%,100% { filter:drop-shadow(0 0 1px #70e3c455); } 50% { filter:drop-shadow(0 0 5px #70e3c4bb); } }
@keyframes arch-signal-flow { to { stroke-dashoffset:-18; } }
@media(max-width:540px) { .picture-choice-grid { grid-template-columns:1fr; gap:9px; } }
@media(prefers-reduced-motion:reduce) {
  .picture-choice-card { transition:none; }
  .picture-choice-card[aria-pressed="true"],.single-run-flow,.arch-node.bright,.arch-memory { animation:none; }
}
""",
    js="""
export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector(".setup-picture-choice");
  if (!root) return;

  root.replaceChildren();
  root.classList.toggle("setup-picture-choice--readonly", Boolean(data.readOnly));
  root.setAttribute("aria-label", data.heading || "Choose an illustrated answer");

  const heading = document.createElement("div");
  heading.className = "picture-choice-heading";
  heading.textContent = data.heading || "";
  root.appendChild(heading);

  const hint = document.createElement("div");
  hint.className = "picture-choice-hint";
  hint.textContent = data.hint || "";
  root.appendChild(hint);

  const grid = document.createElement("div");
  grid.className = "picture-choice-grid";
  for (const item of (data.choices || [])) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "picture-choice-card";
    button.disabled = Boolean(data.readOnly);
    button.setAttribute("aria-pressed", String(data.value === item.value));
    button.setAttribute("aria-label", item.title + ". " + item.description);

    const art = document.createElement("div");
    art.className = "picture-choice-art";
    art.setAttribute("aria-hidden", "true");
    art.innerHTML = item.svg;

    const title = document.createElement("div");
    title.className = "picture-choice-title";
    title.textContent = item.title;

    const description = document.createElement("div");
    description.className = "picture-choice-description";
    description.textContent = item.description;

    button.append(art, title, description);
    button.addEventListener("click", () => {
      for (const other of grid.querySelectorAll(".picture-choice-card")) {
        other.setAttribute("aria-pressed", "false");
      }
      button.setAttribute("aria-pressed", "true");
      setStateValue("value", item.value);
    });
    grid.appendChild(button);
  }
  root.appendChild(grid);
}
""",
)


def _continuous_run_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="One uninterrupted signal continues from start to finish">
  <defs>
    <linearGradient id="single-run-line" x1="0" x2="1"><stop offset="0" stop-color="#58c9b2"/><stop offset="1" stop-color="#a0ead0"/></linearGradient>
    <path id="single-run-path" d="M43 128C91 119 104 89 155 91S216 102 260 78S326 63 367 47S419 34 446 24"/>
  </defs>
  <path d="M42 32H450M42 66H450M42 100H450M42 134H450" stroke="#2a3b39"/>
  <path d="M43 148H451" stroke="#637a75"/>
  <text x="43" y="165" fill="#8ea49e" font-size="10">START</text>
  <text x="416" y="165" fill="#8ea49e" font-size="10">FINISH</text>
  <use href="#single-run-path" fill="none" stroke="url(#single-run-line)" stroke-width="4" stroke-linecap="round"/>
  <use href="#single-run-path" fill="none" stroke="#d3fff1" stroke-width="1.3" stroke-dasharray="8 18" opacity=".62" class="single-run-flow"/>
  <circle cx="43" cy="128" r="4.5" fill="#b1f1dd"/><circle cx="446" cy="24" r="4.5" fill="#b1f1dd"/>
  <text x="240" y="166" fill="#78948c" font-size="9" text-anchor="middle">time keeps moving →</text>
</svg>
'''


def _stacked_runs_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="Three separate signals, each starting at its own beginning">
  <path d="M24 54H455M24 104H455M24 154H455" stroke="#34433f"/>
  <path d="M43 44C84 41 102 20 142 24S194 46 232 36S300 17 353 25S405 31 438 24" fill="none" stroke="#75dfc6" stroke-width="3.4" stroke-linecap="round"/>
  <path d="M43 94C84 91 102 70 142 74S194 96 232 86S300 67 353 75S405 81 438 74" fill="none" stroke="#a5b1ff" stroke-width="3.4" stroke-linecap="round"/>
  <path d="M43 144C84 141 102 120 142 124S194 146 232 136S300 117 353 125S405 131 438 124" fill="none" stroke="#f1c47c" stroke-width="3.4" stroke-linecap="round"/>
  <circle cx="43" cy="44" r="4" fill="#b8f7e6"/><circle cx="43" cy="94" r="4" fill="#ccd2ff"/><circle cx="43" cy="144" r="4" fill="#ffe0a9"/>
  <text x="25" y="15" fill="#bdece0" font-size="9" font-weight="700">RUN 01</text>
  <text x="25" y="65" fill="#c6cdff" font-size="9" font-weight="700">RUN 02</text>
  <text x="25" y="115" fill="#f0d6a5" font-size="9" font-weight="700">RUN 03</text>
  <text x="47" y="169" fill="#839892" font-size="9">t: 0 → T</text>
  <text x="205" y="169" fill="#839892" font-size="9">three separate experiments</text>
  <path d="M452 35v18m0 32v18m0 32v18" stroke="#ffb86b" stroke-width="1.5" stroke-dasharray="3 4"/>
</svg>
'''


def _auto_split_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="Automatically scan a stacked recording and detect where each run restarts">
  <text x="23" y="22" fill="#8ea8a0" font-size="9" font-weight="700" letter-spacing="1">ONE FILE · SEVERAL RUNS</text>
  <path d="M30 52H448M30 91H448M30 130H448" stroke="#2b3b39"/>
  <path d="M36 46C69 43 81 27 111 31S151 51 184 42L188 46C218 43 232 27 262 31S301 51 334 42L338 46C368 43 383 27 412 31S435 39 446 34" fill="none" stroke="#70dbc4" stroke-width="3" stroke-linecap="round"/>
  <path d="M188 27V139M338 27V139" stroke="#ffb86b" stroke-width="1.6" stroke-dasharray="4 5"/>
  <circle cx="188" cy="46" r="4" fill="#ffc47c"/><circle cx="338" cy="46" r="4" fill="#ffc47c"/>
  <text x="62" y="67" fill="#a5cfc3" font-size="9">RUN 1</text>
  <text x="225" y="67" fill="#a5cfc3" font-size="9">RUN 2</text>
  <text x="375" y="67" fill="#a5cfc3" font-size="9">RUN 3</text>
  <rect x="138" y="103" width="204" height="38" rx="10" fill="#172522" stroke="#4b8d7e"/>
  <circle cx="158" cy="122" r="7" fill="none" stroke="#7de0cb" stroke-width="1.5"/>
  <path d="m163 127 5 5" stroke="#7de0cb" stroke-width="1.5" stroke-linecap="round"/>
  <text x="244" y="126" fill="#b9e9dd" font-size="10" font-weight="700" text-anchor="middle">AUTO-DETECT RESTARTS</text>
  <text x="240" y="163" fill="#829890" font-size="9" text-anchor="middle">CHECKS TIME RESETS AND STATE DISCONTINUITIES</text>
</svg>
'''


def _manual_split_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="Enter the exact timestamps where each new trajectory begins">
  <text x="23" y="22" fill="#8ea8a0" font-size="9" font-weight="700" letter-spacing="1">YOU KNOW WHERE RUNS RESTART</text>
  <path d="M35 55H445M35 91H445M35 127H445" stroke="#2b3b39"/>
  <path d="M42 112C72 105 91 68 124 73S167 112 197 99S239 60 270 69S311 112 342 95S387 61 435 75" fill="none" stroke="#70dbc4" stroke-width="3" stroke-linecap="round"/>
  <path d="M168 43V137M326 43V137" stroke="#9fafff" stroke-width="1.6" stroke-dasharray="4 5"/>
  <circle cx="168" cy="101" r="4" fill="#b8c2ff"/><circle cx="326" cy="103" r="4" fill="#b8c2ff"/>
  <rect x="123" y="127" width="90" height="25" rx="7" fill="#202333" stroke="#6977b9"/>
  <rect x="281" y="127" width="90" height="25" rx="7" fill="#202333" stroke="#6977b9"/>
  <text x="168" y="144" fill="#d8ddff" font-size="9" text-anchor="middle">t₁ = 12.5 s</text>
  <text x="326" y="144" fill="#d8ddff" font-size="9" text-anchor="middle">t₂ = 25.0 s</text>
  <text x="240" y="169" fill="#829890" font-size="9" text-anchor="middle">ENTER THE START TIME OF EACH NEW RUN</text>
</svg>
'''


def _no_wrap_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="A signal passes smoothly through the plus pi boundary without a jump">
  <path d="M52 25H454M52 67H454M52 109H454M52 151H454" stroke="#2d3d3b"/>
  <path d="M52 67H454" stroke="#657c76" stroke-dasharray="4 5"/>
  <text x="13" y="71" fill="#a7bbb4" font-size="10">+π</text>
  <text x="28" y="113" fill="#718781" font-size="10">0</text>
  <text x="52" y="169" fill="#819891" font-size="9">TIME →</text>
  <path d="M54 139C114 133 142 113 196 105S267 85 305 65S375 46 451 33" fill="none" stroke="#70dbc4" stroke-width="4" stroke-linecap="round"/>
  <circle cx="305" cy="65" r="5" fill="#b8f7e6"/>
  <text x="322" y="81" fill="#98cdbc" font-size="10">smooth through +π</text>
</svg>
'''


def _wrap_jump_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="A stored angle jumps from plus pi to minus pi at the wrap boundary">
  <path d="M52 25H454M52 67H454M52 109H454M52 151H454" stroke="#2d3d3b"/>
  <path d="M52 67H454M52 151H454" stroke="#657c76" stroke-dasharray="4 5"/>
  <text x="13" y="71" fill="#a7bbb4" font-size="10">+π</text>
  <text x="13" y="155" fill="#a7bbb4" font-size="10">−π</text>
  <text x="52" y="169" fill="#819891" font-size="9">TIME →</text>
  <path d="M54 130C104 124 143 107 190 91S244 73 279 67L282 151C329 143 373 129 451 113" fill="none" stroke="#70dbc4" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="279" cy="67" r="5" fill="#ffc47c"/><circle cx="282" cy="151" r="5" fill="#ffc47c"/>
  <path d="M299 78V140" stroke="#ffb86b" stroke-width="1.7" stroke-dasharray="4 4"/>
  <text x="311" y="111" fill="#ffc98e" font-size="10">wrap</text>
</svg>
'''


def _manual_angle_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="Choose the exact angle state columns yourself">
  <text x="24" y="25" fill="#8ea8a0" font-size="9" font-weight="700" letter-spacing="1">PICK THE STATE COLUMNS</text>
  <rect x="22" y="39" width="436" height="34" rx="8" fill="#172522" stroke="#4e9c89"/>
  <rect x="35" y="50" width="13" height="13" rx="3" fill="#61d3b9" stroke="#b4f5e7"/>
  <path d="m38 56 3 3 5-6" fill="none" stroke="#12302a" stroke-width="2" stroke-linecap="round"/>
  <text x="62" y="60" fill="#ddf4ed" font-size="12" font-family="ui-monospace,monospace">state_1</text>
  <text x="425" y="60" fill="#7ce0c9" font-size="9" text-anchor="end">SELECTED</text>
  <rect x="22" y="81" width="436" height="34" rx="8" fill="#141a1c" stroke="#2d403d"/>
  <rect x="35" y="92" width="13" height="13" rx="3" fill="#101716" stroke="#526761"/>
  <text x="62" y="102" fill="#b5c3be" font-size="12" font-family="ui-monospace,monospace">state_2</text>
  <rect x="22" y="123" width="436" height="34" rx="8" fill="#141a1c" stroke="#2d403d"/>
  <rect x="35" y="134" width="13" height="13" rx="3" fill="#101716" stroke="#526761"/>
  <text x="62" y="144" fill="#b5c3be" font-size="12" font-family="ui-monospace,monospace">state_3</text>
  <text x="455" y="169" fill="#829890" font-size="9" text-anchor="end">YOU CHOOSE THE EXACT ANGLE STATE(S)</text>
</svg>
'''


def _auto_angle_svg() -> str:
    return '''
<svg viewBox="0 0 480 176" role="img" aria-label="Automatically scan measured signals for a plus pi to minus pi jump">
  <path d="M35 40H328M35 82H328M35 124H328" stroke="#2b3b39"/>
  <path d="M35 82H328" stroke="#657c76" stroke-dasharray="4 5"/>
  <text x="8" y="86" fill="#91aaa2" font-size="9">+π</text>
  <text x="10" y="128" fill="#91aaa2" font-size="9">−π</text>
  <path d="M38 111C80 107 112 91 150 87S205 78 235 82L238 124C264 120 292 111 326 104" fill="none" stroke="#70dbc4" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="235" cy="82" r="4" fill="#ffc47c"/><circle cx="238" cy="124" r="4" fill="#ffc47c"/>
  <path d="M236 57V72M230 66l6 6 6-6" fill="none" stroke="#ffc47c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="357" y="48" width="104" height="72" rx="11" fill="#172522" stroke="#4b8d7e"/>
  <circle cx="377" cy="68" r="7" fill="none" stroke="#7de0cb" stroke-width="1.5"/>
  <path d="m382 73 5 5" stroke="#7de0cb" stroke-width="1.5" stroke-linecap="round"/>
  <text x="409" y="73" fill="#b9e9dd" font-size="9" font-weight="700" text-anchor="middle">AUTO SCAN</text>
  <text x="409" y="94" fill="#82a69b" font-size="8" text-anchor="middle">finds ±π jumps</text>
  <path d="M329 84H350" stroke="#68cdb4" stroke-width="1.5" stroke-dasharray="3 3"/>
  <text x="35" y="157" fill="#829890" font-size="9">LOADER CHECKS THE SIGNAL WHEN TRAINING STARTS</text>
</svg>
'''


def _pinn_explainer_svg() -> str:
    """A compact, illustrative PINN diagram for the setup question."""
    return '''
<svg viewBox="0 0 520 126" role="img" aria-label="A PINN compares measured data and a physical equation while learning system behavior">
  <path d="M18 26H502M18 100H502" stroke="#263936" stroke-width="1"/>
  <rect x="18" y="39" width="126" height="52" rx="11" fill="#172522" stroke="#3c6d60"/>
  <text x="81" y="58" text-anchor="middle" fill="#8ce3cf" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">MEASUREMENTS</text>
  <path d="M36 78C49 77 49 65 62 65S76 83 89 77S104 58 119 62" fill="none" stroke="#70dbc4" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M145 65H189" fill="none" stroke="#6ccfba" stroke-width="1.5" stroke-dasharray="4 5"/>
  <rect x="190" y="39" width="126" height="52" rx="11" fill="#1d2028" stroke="#625c83"/>
  <text x="253" y="58" text-anchor="middle" fill="#c8c2ff" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">PHYSICS EQUATION</text>
  <text x="253" y="76" text-anchor="middle" fill="#d5d8df" font-size="8.5" font-family="ui-monospace,monospace">rate = f(state, input)</text>
  <path d="M317 65H357" fill="none" stroke="#6ccfba" stroke-width="1.5" stroke-dasharray="4 5"/>
  <rect x="358" y="39" width="144" height="52" rx="11" fill="#17202a" stroke="#4c7695"/>
  <text x="430" y="58" text-anchor="middle" fill="#a9d7f0" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">PINN MODEL</text>
  <circle cx="390" cy="76" r="4" fill="#70dbc4"><animate attributeName="opacity" values=".35;1;.35" dur="1.8s" repeatCount="indefinite"/></circle>
  <circle cx="430" cy="76" r="4" fill="#70dbc4"><animate attributeName="opacity" values="1;.35;1" dur="1.8s" repeatCount="indefinite"/></circle>
  <circle cx="470" cy="76" r="4" fill="#70dbc4"><animate attributeName="opacity" values=".35;1;.35" dur="1.8s" repeatCount="indefinite"/></circle>
  <text x="260" y="117" text-anchor="middle" fill="#91aaa2" font-size="9" font-family="ui-monospace,monospace">FITS THE DATA WHILE PENALIZING EQUATION MISMATCH</text>
</svg>
'''


def _pcnn_explainer_svg() -> str:
    """Diagram the PCNN's exact-output path separately from PINN's soft loss."""
    return '''
<svg viewBox="0 0 520 126" role="img" aria-label="PCNN routes selected state derivatives through exact rules and learns remaining states with a neural network">
  <path d="M18 26H502M18 100H502" stroke="#263936" stroke-width="1"/>
  <rect x="18" y="39" width="142" height="52" rx="11" fill="#172522" stroke="#3c6d60"/>
  <text x="89" y="58" text-anchor="middle" fill="#8ce3cf" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">EXACT RULE</text>
  <text x="89" y="77" text-anchor="middle" fill="#d5d8df" font-size="8.5" font-family="ui-monospace,monospace">ẋⱼ = fⱼ(x,u)</text>
  <path d="M161 65H204" fill="none" stroke="#6ccfba" stroke-width="1.5" stroke-dasharray="4 5"/>
  <rect x="205" y="39" width="126" height="52" rx="11" fill="#1d2028" stroke="#625c83"/>
  <text x="268" y="58" text-anchor="middle" fill="#c8c2ff" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">SELECTED STATE</text>
  <text x="268" y="77" text-anchor="middle" fill="#d5d8df" font-size="8.5" font-family="ui-monospace,monospace">exact output</text>
  <path d="M332 65H365" fill="none" stroke="#6ccfba" stroke-width="1.5" stroke-dasharray="4 5"/>
  <rect x="366" y="39" width="136" height="52" rx="11" fill="#17202a" stroke="#4c7695"/>
  <text x="434" y="58" text-anchor="middle" fill="#a9d7f0" font-size="9" font-family="ui-monospace,monospace" letter-spacing="1">OTHER STATES</text>
  <text x="434" y="77" text-anchor="middle" fill="#d5d8df" font-size="8.5" font-family="ui-monospace,monospace">learned by NN</text>
  <text x="260" y="117" text-anchor="middle" fill="#91aaa2" font-size="9" font-family="ui-monospace,monospace">SELECTED RULES REPLACE NN OUTPUTS EXACTLY</text>
</svg>
'''


def _physics_choice_svg(label: str) -> str:
    labels = {"data": "MEASURED DATA", "pinn": "MEASUREMENTS + PHYSICS", "pcnn": "HARD RULE + NN", "source": "CONSTRAINT SOURCE"}
    title = labels.get(label, "SYSTEM MODEL")
    return f'''<svg viewBox="0 0 480 150" role="img" aria-label="{title.title()}">
      <rect x="24" y="22" width="432" height="106" rx="14" fill="#151e1d" stroke="#3a544e"/>
      <path d="M48 92H432M48 62H432" stroke="#263936"/>
      <path d="M50 88C85 87 91 42 125 48S165 99 198 84S240 47 274 59S315 89 347 74S393 50 430 57" fill="none" stroke="#70dbc4" stroke-width="3" stroke-linecap="round"/>
      <text x="48" y="43" fill="#a7d7cb" font-size="10" font-family="ui-monospace,monospace" letter-spacing="1">{title}</text>
      <circle cx="274" cy="59" r="5" fill="#8ee1cd"><animate attributeName="r" values="4;7;4" dur="2s" repeatCount="indefinite"/></circle>
    </svg>'''


def render_picture_choice(
    *,
    key: str,
    heading: str,
    hint: str,
    choices: list[dict],
    initial_value: str | None = None,
    confirm: bool = True,
    read_only: bool = False,
) -> str | None:
    """Render selectable illustrated choices and optionally ask for confirmation."""
    component_key = f"setup_picture_choice_{key}"
    current = st.session_state.get(component_key, {})
    selected = current.get("value", initial_value) if isinstance(current, dict) else initial_value
    result = _PICTURE_CHOICE(
        key=component_key,
        data={"heading": heading, "hint": hint, "choices": choices, "value": selected,
              "readOnly": read_only},
        default={"value": selected},
        on_value_change=lambda: None,
        width="stretch",
        height="content",
    )
    selected = result.value
    selected_choice = next((item for item in choices if item["value"] == selected), None)
    if not selected_choice:
        return None
    if confirm:
        if st.button(
            f"Continue with {selected_choice['title']}",
            key=f"setup_picture_continue_{key}",
            type="primary",
        ):
            st.session_state["conversation_pending"] = selected_choice["value"]
    return selected_choice["value"]


def _cycle_history(entry: dict) -> pd.DataFrame:
    """Read finite per-candidate scores saved in the manifest or activity feed."""
    rows = entry.get("performance_history") or []
    if not rows and entry.get("run_dir"):
        rows = [row.get("details", {}) for row in activity.load(entry["run_dir"])
                if row.get("id", "").startswith("actor:") and row.get("state") == "complete"]
    clean = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        try:
            cycle = int(row.get("cycle", row.get("iteration", index)))
            train = float(row["train_mse"])
            validation = float(row.get("val_mse", row.get("mse")))
            if not (math.isfinite(train) and math.isfinite(validation)):
                continue
            clean.append({"Cycle": cycle, "Training MSE": train,
                          "Validation MSE": validation,
                          "Training seconds": float(row.get("training_seconds") or 0),
                          "Configuration": row.get("config") if isinstance(row.get("config"), dict) else {}})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return pd.DataFrame(clean).sort_values("Cycle") if clean else pd.DataFrame()


def _best_nn_summary(entry: dict, history: pd.DataFrame) -> str:
    """Describe the winning network and its predicted output in one short line."""
    architecture = str(entry.get("architecture") or "neural network").strip().upper()
    config = entry.get("best_config")
    if not isinstance(config, dict) or not config:
        if not history.empty:
            config = history.loc[history["Validation MSE"].idxmin()].get("Configuration")
    config = config if isinstance(config, dict) else {}
    widths = []
    for value in config.get("hidden_layers") or []:
        try:
            width = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if width > 0:
            widths.append(width)
    topology = f" ({len(widths)} layers, {' → '.join(map(str, widths))})" if widths else ""
    try:
        state_count = int(entry.get("state_dim"))
    except (TypeError, ValueError, OverflowError):
        state_count = 0
    outputs = f"{state_count} state derivatives (xdot)" if state_count > 0 else "state derivatives (xdot)"
    physics_mode = str(entry.get("physics_mode") or "pinn").upper()
    model_type = f"{physics_mode}-{architecture}" if entry.get("use_pinn") else architecture
    return f"Best NN · {model_type}{topology} · predicts {outputs}."


def _plot_label(path: str) -> str:
    stem = Path(path).stem.lower()
    for token, label in [("test_verification", "Held-out predictions"),
                         ("mse_convergence", "Validation MSE by cycle"),
                         ("rmse_convergence", "Validation RMSE by cycle"),
                         ("latency_evolution", "Inference latency"),
                         ("hyperparameters", "Training settings explored"),
                         ("layers_neurons", "Architecture search")]:
        if token in stem:
            return label
    return Path(path).stem.replace("_", " ").replace("-", " ").title()


def _cycle_curve(history: pd.DataFrame):
    if history.empty:
        st.caption("Per-cycle scores were not saved for this run.")
        return
    chart = history.set_index("Cycle")[["Training MSE", "Validation MSE"]]
    st.line_chart(chart, height=230, y_label="Mean squared error")
    st.caption("Each point is one candidate model. Lower validation error is better; training and validation gaps can reveal overfitting.")


def _cycle_inspector(history: pd.DataFrame, key: str):
    """Show the selected candidate's scores and saved configuration."""
    if history.empty:
        st.info("Candidate cycle details were not saved for this run.")
        return

    cycles = sorted(int(value) for value in history["Cycle"].dropna().unique())
    st.markdown("#### Candidate cycle settings")
    st.caption("Select a cycle to review its scores, training time, and configuration.")
    if len(cycles) == 1:
        cycle = cycles[0]
        st.caption(f"Only candidate cycle recorded: {cycle}.")
    else:
        cycle = st.select_slider("Candidate cycle", options=cycles,
                                 value=cycles[0], key=f"inspect_cycle_{key}")

    row = history.loc[history["Cycle"] == cycle].iloc[-1]
    a, b, c = st.columns(3)
    a.metric("Training MSE", f"{row['Training MSE']:.4g}")
    b.metric("Validation MSE", f"{row['Validation MSE']:.4g}")
    training_seconds = row["Training seconds"]
    c.metric("Training time", f"{training_seconds:.1f} s" if _finite(training_seconds) else "—")
    configuration = row["Configuration"]
    if configuration:
        with st.expander(f"Cycle {cycle} · candidate settings", expanded=True):
            st.json(configuration)
    else:
        st.caption("No configuration details were saved for this cycle.")


def render_run_result(entry: dict, key: str, requested_cycles: int | None = None):
    """Show a compact, interactive evidence dashboard directly in the chat."""
    status = str(entry.get("status") or "unknown").replace("_", " ").title()
    history = _cycle_history(entry)
    actual_cycles = int(entry.get("cycles_run") or (int(history["Cycle"].max()) if not history.empty else 0))
    requested = max(1, int(requested_cycles or actual_cycles or 1))
    best_mse = entry.get("best_mse")
    best_rmse = entry.get("best_rmse")
    latency = entry.get("latency_ms")
    with st.container(border=True, key=f"run_dashboard_{key}"):
        st.markdown(f"#### :material/monitoring: Run snapshot · {status}")
        a, b, c, d, e = st.columns(5)
        a.metric("Best validation MSE", f"{float(best_mse):.4g}" if best_mse is not None and _finite(best_mse) else "—")
        b.metric("Best validation RMSE", f"{float(best_rmse):.4g}" if best_rmse is not None and _finite(best_rmse) else "—")
        c.metric("Inference latency", f"{float(latency):.3g} ms" if latency is not None and _finite(latency) else "—")
        d.metric("Search cycles", f"{actual_cycles} / {requested}")
        score = entry.get("success_score")
        e.metric("Fit + rollout score", f"{float(score):.1f} / 100" if score is not None and _finite(score) else "Unavailable",
                 help="Validation fit and held-out rollout heuristic; not a stability or safety guarantee.")
        st.caption(_best_nn_summary(entry, history))
        training_seconds = entry.get("training_seconds")
        elapsed_seconds = entry.get("elapsed_seconds")
        note = []
        if training_seconds is not None and _finite(training_seconds):
            note.append(f"{float(training_seconds):.1f} s model training")
        if elapsed_seconds is not None and _finite(elapsed_seconds):
            note.append(f"{float(elapsed_seconds):.1f} s total run time")
        if entry.get("run_mode"):
            note.append(f"{str(entry['run_mode']).title()} search")
        st.caption("  ·  ".join(note) if note else "Explore the run results below.")
        estimates = entry.get("physics_parameter_estimates") or {}
        if estimates:
            metadata = {item.get("alias"): item for item in (entry.get("physics_parameter_metadata") or [])}
            with st.expander("Estimated physics coefficients", expanded=False):
                rows = []
                for alias, value in estimates.items():
                    item = metadata.get(alias, {})
                    rows.append({
                        "Coefficient": item.get("name") or alias,
                        "Estimate": value,
                        "Allowed range": (f"{item.get('lower')} to {item.get('upper')}"
                                          if item.get("lower") is not None and item.get("upper") is not None else "—"),
                        "Units": item.get("units") or "—",
                    })
                st.dataframe(rows, width="stretch", hide_index=True)
                st.caption("These coefficients were estimated during training within the reviewed bounds. Their physical accuracy depends on the model assumptions and the data's identifiability.")
        with st.expander("Performance by cycle", expanded=False):
            _cycle_curve(history)
        with st.expander("State accuracy", expanded=False):
            comparison = analysis.compare_run_states(entry.get("run_dir", ""))
            if comparison.get("rows"):
                render_state_comparison(comparison)
            else:
                st.info("Per-state held-out scores were not saved for this run.")


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def render_answer_visuals(question: str, key: str, *, dataset: dict | None = None,
                          run_dir: str | None = None):
    """Attach useful plots to a model answer, choosing evidence from this run or dataset."""
    if run_dir:
        entry = {"run_dir": run_dir}
        manifest_path = Path(run_dir) / "run_manifest.json"
        try:
            import json
            entry.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        entry["run_dir"] = run_dir
        history = _cycle_history(entry)
        comparison = analysis.compare_run_states(run_dir)
        plots = hist.figures_of(entry)
        q = question.casefold()
        preferred = "State accuracy" if "state" in q or re.search(r"which .*predict.*(best|better|worst)|state.*(best|better|worst)", q) else (
            "Performance" if any(word in q for word in ("cycle", "training", "converg", "overfit", "fail", "score", "validation")) else "Run plots")
        with st.expander("Visuals for this answer", icon=":material/insights:"):
            choices = ["Performance"]
            if comparison.get("rows"):
                choices.append("State accuracy")
            if plots:
                choices.append("Saved plots")
            selection = st.segmented_control("View", choices, default=preferred if preferred in choices else choices[0],
                                             key=f"answer_visual_{key}", label_visibility="collapsed")
            if selection == "Performance":
                _cycle_curve(history)
            elif selection == "State accuracy":
                render_state_comparison(comparison)
            elif plots:
                selected = st.selectbox("Relevant run image", plots, format_func=_plot_label,
                                        key=f"answer_plot_{key}")
                st.image(selected, width="stretch")
    elif dataset:
        profile = analysis.ensure_profile(dataset)
        _render_dataset_evidence(dataset, key, profile)


def _render_dataset_evidence(dataset: dict, key: str, profile: dict | None = None) -> None:
    """Show the dataset's signal, distribution, and input/output views in one place."""
    profile = profile or analysis.ensure_profile(dataset)
    states = [item["name"] for item in profile.get("states", [])]
    inputs = [item["name"] for item in profile.get("inputs", [])]
    if not states:
        return

    with st.expander("Visual evidence from your data", icon=":material/insights:", expanded=False):
        options = ["Min and max", "Distribution", "Input vs output"]
        selection = st.segmented_control(
            "Dataset view", options, default="Min and max",
            key=f"dataset_evidence_view_{key}", label_visibility="collapsed", wrap=False,
        )
        try:
            frame = _preview(dataset["path"], dataset["sha256"], dataset["name"])
        except (OSError, ValueError) as exc:
            st.warning(str(exc))
            return

        if selection == "Min and max":
            selected = st.selectbox(
                "State to plot", states, key=f"data_evidence_state_{key}",
            )
            chart_columns = (["time"] if "time" in frame else []) + [selected]
            chart = frame[chart_columns].copy()
            chart[selected] = pd.to_numeric(chart[selected], errors="coerce")
            if "time" in chart:
                st.line_chart(chart, x="time", y=selected, x_label="Time (s)", height=245)
            else:
                st.line_chart(chart, y=selected, x_label="Sample", height=245)

            rows = []
            for item in profile.get("states", []):
                rows.append({
                    "State": item["name"], "Minimum": item["min"], "Maximum": item["max"],
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                             height=min(260, 42 + 35 * len(rows)))
                st.caption("Ranges are measured from the uploaded data; they are not model scores.")
        elif selection == "Distribution":
            name = st.selectbox("State distribution", states, key=f"data_evidence_distribution_{key}")
            try:
                values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy()
                if len(values):
                    counts, edges = np.histogram(values, bins=min(32, max(8, int(np.sqrt(len(values))))))
                    dist = pd.DataFrame({
                        "Value range": [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(counts))],
                        "Samples": counts,
                    })
                    st.bar_chart(dist, x="Value range", y="Samples", height=225)
            except (ValueError, KeyError) as exc:
                st.warning(str(exc))
        elif selection == "Input vs output":
            if not inputs:
                st.caption("No input signals are assigned in this dataset.")
                return
            a, b = st.columns(2)
            input_name = a.selectbox("Input", inputs, key=f"data_evidence_input_{key}")
            output_name = b.selectbox("Output", states, key=f"data_evidence_output_{key}")
            try:
                points = frame[[input_name, output_name]].apply(pd.to_numeric, errors="coerce").dropna()
                st.scatter_chart(points, x=input_name, y=output_name, height=245)
                if len(points) > 1:
                    corr = points[input_name].corr(points[output_name])
                    st.caption(f"Correlation in the recording: {corr:.3f}. Correlation alone does not establish causation or identify dynamics.")
            except (ValueError, KeyError) as exc:
                st.warning(str(exc))


def render_data_quality_notice(dataset: dict, key: str) -> None:
    """Put the most actionable data-profile advisory before the review prose."""
    profile = analysis.ensure_profile(dataset)
    pairs = []
    for pair in profile.get("correlated_state_pairs", []):
        try:
            correlation = float(pair.get("abs_correlation"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(correlation) and correlation >= 0.99:
            pairs.append((str(pair.get("state_a", "State A")),
                          str(pair.get("state_b", "State B")), correlation))
    if not pairs:
        return

    pairs.sort(key=lambda item: item[2], reverse=True)
    with st.container(border=True, key=f"data_quality_notice_{key}"):
        left, right = st.columns([1, 5])
        with left:
            st.badge("Review", icon=":material/warning:", color="orange")
        with right:
            st.markdown("**Two state signals are almost perfectly correlated**")
            st.caption(
                "Check whether these columns represent distinct measurements. "
                "Near-duplicate signals can make their separate behavior difficult to identify."
            )
        for state_a, state_b, correlation in pairs[:3]:
            # st.text renders dataset-provided column names literally.
            st.text(f"{state_a}  ↔  {state_b}   ·   |r| = {correlation:.6f}")
        if len(pairs) > 3:
            st.caption(f"And {len(pairs) - 3} more highly correlated state pairs")


def render_dataset(dataset: dict, key: str):
    """Show measured evidence in the message, without implying model performance."""
    profile = analysis.ensure_profile(dataset)
    interval = profile.get("median_dt")
    duration = profile.get("observed_duration")
    with st.container(border=True, key=f"data_review_{key}"):
        st.caption("MEASURED DATA · BEFORE MODEL TRAINING")
        a, b, c = st.columns(3)
        a.metric("Samples", f"{profile['rows']:,}")
        b.metric("Observed time", f"{duration:.3g} s" if duration is not None else "Unknown")
        c.metric("Median interval", f"{interval:.4g} s" if interval is not None else "Unknown")
        _render_dataset_evidence(dataset, key, profile)
        if profile.get("inputs"):
            varying = sum(not item["constant"] for item in profile["inputs"])
            st.caption(f"Inputs varying: {varying}/{len(profile['inputs'])} · Variation alone does not prove adequate excitation.")


def _choose(label: str, value: str, key: str, *, primary=False):
    if st.button(label, key=key, type="primary" if primary else "secondary"):
        st.session_state["conversation_pending"] = value


def _trajectory_choices():
    return [
        {
            "title": "One continuous run",
            "description": "One experiment from start to finish; time never resets.",
            "value": "One continuous run",
            "svg": _continuous_run_svg(),
        },
        {
            "title": "Several stacked runs",
            "description": "Separate experiments share one file; each run starts again.",
            "value": "Several stacked runs",
            "svg": _stacked_runs_svg(),
        },
    ]


def _split_method_choices():
    return [
        {
            "title": "Auto-detect boundaries",
            "description": "LabCD scans for time resets and state jumps between stacked runs.",
            "value": "Detect boundaries",
            "svg": _auto_split_svg(),
        },
        {
            "title": "Set split times manually",
            "description": "Enter the timestamp where each new experiment begins.",
            "value": "I know split timestamps",
            "svg": _manual_split_svg(),
        },
    ]


def render_setup_question(chat: dict, key: str, active: bool, question_timer=None):
    if not active:
        return
    stage = (chat.get("setup") or {}).get("stage")
    dataset = chat.get("dataset") or {}
    profile = dataset.get("analysis") or {}
    if dataset.get("path") and dataset.get("states") and "boundary_events" not in profile:
        try:
            profile = analysis.ensure_profile(dataset)
        except (OSError, ValueError, KeyError):
            pass
    with st.container(border=True, key=f"setup_{key}"):
        st.caption("DATA ASSUMPTIONS · ANSWER HERE OR TYPE IN CHAT")
        if stage == "trajectory":
            boundary_events = profile.get("boundary_events") or []
            if boundary_events:
                event_count = int(profile.get("boundary_event_count", len(boundary_events)) or len(boundary_events))
                table_rows = []
                for event in boundary_events:
                    evidence = []
                    if event.get("time_reset"):
                        evidence.append("Time reset")
                    if event.get("state_jumps"):
                        evidence.append("Large state jump · " + ", ".join(event["state_jumps"]))
                    before, after = event.get("time_before"), event.get("time_after")
                    table_rows.append([
                        f"Row {int(event.get('data_row', 0)):,}",
                        f"{before:.5g}" if before is not None else "—",
                        f"{after:.5g}" if after is not None else "—",
                        " + ".join(evidence) or "Unclassified signal change",
                    ])
                note = ("These are pre-scan candidates, not proof of separate experiments. A real system response can also create a large jump; confirm how the data was recorded.")
                if event_count > len(table_rows):
                    note += f" Showing the first {len(table_rows)} of {event_count} candidates."
                reset_count = int(profile.get("time_reset_count", 0) or 0)
                jump_count = int(profile.get("possible_boundary_count", 0) or 0)
                _render_detection_table(
                    "Possible run boundaries detected",
                    f"Pre-scan found {reset_count} time reset(s) and {jump_count} large state-step candidate(s).",
                    f"{event_count} candidate{'s' if event_count != 1 else ''}",
                    ["Data location", "Time before (s)", "Time after (s)", "Detection cues"],
                    table_rows,
                    note,
                )
            selected_trajectory = render_picture_choice(
                key=f"trajectory_{key}",
                heading="Which picture matches your data?",
                hint="A trajectory is one uninterrupted recording. Click the picture that matches; I’ll move to the next question.",
                choices=_trajectory_choices(),
                confirm=False,
            )
            if selected_trajectory:
                st.session_state["conversation_pending"] = selected_trajectory
        elif stage == "split":
            render_picture_choice(
                key=f"trajectory_context_{key}",
                heading="Which picture matches your data?",
                hint="You chose several stacked runs. Now choose how to locate their boundaries.",
                choices=_trajectory_choices(),
                initial_value="Several stacked runs",
                confirm=False,
                read_only=True,
            )
            selected_split = render_picture_choice(
                key=f"split_method_{key}",
                heading="How should I locate the run boundaries?",
                hint="Choose the picture that matches what you know about where each experiment starts.",
                choices=_split_method_choices(),
                confirm=False,
            )
            if selected_split:
                st.session_state["conversation_pending"] = selected_split
        elif stage == "split_times":
            with st.form(f"split_times_{key}", border=False):
                times = st.text_input("New-run timestamps (seconds)", placeholder="12.5, 25.0")
                if st.form_submit_button("Use these timestamps", type="primary"):
                    st.session_state["conversation_pending"] = f"Split times: {times}"
        elif stage == "angle":
            found = profile.get("detected_wrap_states") or []
            names = profile.get("angle_named_states") or []
            if found:
                signal_by_name = {item.get("name"): item for item in profile.get("states", [])}
                wrap_rows = []
                for state in found:
                    signal = signal_by_name.get(state, {})
                    minimum, maximum = signal.get("min"), signal.get("max")
                    observed_range = (f"{minimum:.4g} to {maximum:.4g}" if minimum is not None and maximum is not None else "—")
                    wrap_rows.append([
                        state,
                        str(int(signal.get("wrap_jumps", 0) or 0)),
                        observed_range,
                    ])
                _render_detection_table(
                    "Possible angle wraps detected",
                    "These state columns contain abrupt jumps near the ±π limits.",
                    f"{len(found)} state{'s' if len(found) != 1 else ''}",
                    ["Measured state", "Wrap-like jumps", "Observed range (rad)"],
                    wrap_rows,
                    "A jump alone does not prove a column is an angle. Confirm it is stored in radians modulo 2π before enabling angle wrapping.",
                )
                st.markdown("**Signal preview for the first candidate**")
                _signal(dataset, found[0], f"angle_{key}", height=165)
            elif names:
                st.markdown("**Angle-like column in your data**")
                _signal(dataset, names[0], f"angle_{key}", height=165)
                candidates = ", ".join(names[:8])
                extra = f" Possible angle-like columns: {candidates}." if candidates else ""
                st.caption("The name suggests an angle, but this recording shows no ±π crossing. Confirm the physical meaning before selecting a state." + extra +
                           " A rate or derivative should not be selected just because its name contains pitch or yaw.")
            else:
                st.caption("I did not identify a likely wrapped angle in this file. This illustration shows what wrapping would look like.")
            wrap_value = "Wrap detected states" if found else "Yes"
            wrap_title = "Yes · use detected states" if found else "Yes · an angle wraps"
            wrap_description = ("Use the state column(s) where the ±π jump was detected."
                                if found else "A stored angle jumps from +π to −π when it crosses the boundary.")
            selected_wrap = render_picture_choice(
                key=f"angle_{key}",
                heading="Does any state wrap at ±π?",
                hint="Choose Yes only for an angle in radians stored modulo 2π. If you choose Yes, select how to identify the angle states.",
                choices=[
                    {
                        "title": wrap_title,
                        "description": wrap_description,
                        "value": wrap_value,
                        "svg": _wrap_jump_svg(),
                    },
                    {
                        "title": "No · no wrapped angles",
                        "description": "The recorded state does not jump from +π to −π.",
                        "value": "No wrapped states",
                        "svg": _no_wrap_svg(),
                    },
                ],
                confirm=False,
            )
            if selected_wrap == "No wrapped states":
                st.session_state["conversation_pending"] = selected_wrap
            elif selected_wrap in ("Yes", "Wrap detected states"):
                method = render_picture_choice(
                    key=f"angle_method_{key}",
                    heading="How should LabCD identify the wrapped angle states?",
                    hint="Choose a picture. Manual lets you select the exact columns; auto-detect scans the measured signals.",
                    choices=[
                        {
                            "title": "Choose states manually",
                            "description": "You select the exact state column or columns that represent wrapped angles.",
                            "value": "Choose angle states",
                            "svg": _manual_angle_svg(),
                        },
                        {
                            "title": "Auto-detect during run",
                            "description": "The loader scans the signals for ±π jumps when training starts.",
                            "value": "Auto-detect wrapped angles",
                            "svg": _auto_angle_svg(),
                        },
                    ],
                    confirm=False,
                )
                if method:
                    st.session_state["conversation_pending"] = method
        elif stage == "angle_select":
            states = dataset.get("states") or []
            selected = st.multiselect("Angle states in radians that wrap at ±π", states, key=f"angle_names_{key}")
            if st.button("Use selected states", type="primary", disabled=not selected, key=f"confirm_angles_{key}"):
                st.session_state["conversation_pending"] = "Wrap states: " + ", ".join(selected)
        elif stage == "pinn_interest":
            with st.container(border=True, key=f"setup_pinn_panel_{key}"):
                st.markdown(_pinn_explainer_svg(), unsafe_allow_html=True)
                st.caption("PINN fits the measurements while penalizing mismatch with a reviewed system equation.")
                st.markdown("**Would you like to use PINN?**")
                yes_col, no_col = st.columns(2, gap="small")
                if yes_col.button("Yes · use PINN", icon=":material/science:", type="primary", key=f"pinn_yes_{key}", width="stretch"):
                    st.session_state["conversation_pending"] = "Yes, use PINN"
                if no_col.button("No · continue", icon=":material/arrow_forward:", key=f"pinn_no_{key}", width="stretch"):
                    st.session_state["conversation_pending"] = "No PINN"
        elif stage == "pcnn_interest":
            with st.container(border=True, key=f"setup_pcnn_panel_{key}"):
                st.markdown(_pcnn_explainer_svg(), unsafe_allow_html=True)
                st.caption("PCNN replaces selected state derivatives with exact hard constraints. The NN learns all other states directly from measured data.")
                st.markdown("**Would you like to use PCNN?**")
                yes_col, no_col = st.columns(2, gap="small")
                if yes_col.button("Yes · use PCNN", icon=":material/account_tree:", type="primary", key=f"pcnn_yes_{key}", width="stretch"):
                    st.session_state["conversation_pending"] = "Yes, use PCNN"
                if no_col.button("No · continue", icon=":material/arrow_forward:", key=f"pcnn_no_{key}", width="stretch"):
                    st.session_state["conversation_pending"] = "No PCNN"
        elif stage == "physics_source_choice":
            pcnn_mode = str((chat.get("setup") or {}).get("pending_physics_mode", "pinn")).lower() == "pcnn"
            selected = render_picture_choice(
                key=f"physics_source_choice_{key}",
                heading="How should we define the hard constraint?" if pcnn_mode else "How should we get the equation?",
                hint=("Define only the state-derivative rule(s) PCNN should enforce exactly. Other states stay neural-network outputs."
                      if pcnn_mode else "PINN needs a reviewed derivative model for the measured states."),
                choices=[
                    {"title": "I have a constraint/source" if pcnn_mode else "I have an equation/source",
                     "description": ("Attach or paste the exact rule(s) to enforce; you can select states later."
                                     if pcnn_mode else "Attach a .py, .m, or .txt source, or paste the equations."),
                     "value": "I will provide my hard constraint source" if pcnn_mode else "I will provide my equation source",
                     "svg": _physics_choice_svg("source")},
                    {"title": "Have LabCD suggest constraints" if pcnn_mode else "Have LabCD draft it",
                     "description": ("Describe the system and exact relationships you trust; review each proposed constraint."
                                     if pcnn_mode else "Describe the model and review the equations and code LabCD proposes."),
                     "value": "LabCD, draft hard constraints from my description" if pcnn_mode else "LabCD, draft it from my description",
                     "svg": _pcnn_explainer_svg() if pcnn_mode else _pinn_explainer_svg()},
                ], confirm=False,
            )
            if selected:
                st.session_state["conversation_pending"] = selected
        elif stage == "physics_source_upload":
            pcnn_mode = str((chat.get("setup") or {}).get("pending_physics_mode", "pinn")).lower() == "pcnn"
            st.caption(
                "Add only the state-derivative rule(s) you want PCNN to enforce exactly. Leave other states out; the neural network learns them."
                if pcnn_mode else "Choose a source file or paste one derivative equation for each measured state."
            )
            with st.form(f"physics_source_form_{key}", border=False):
                uploaded_source = st.file_uploader(
                    "Hard-constraint source · Python, MATLAB, or text" if pcnn_mode else "Equation source · Python, MATLAB, or text",
                    type=["py", "m", "txt"],
                    key=f"physics_source_file_{key}",
                )
                equation_text = st.text_area(
                    "Or paste the exact state-derivative rule(s)" if pcnn_mode else "Or paste the state equations",
                    max_chars=12000,
                    placeholder=("For example: d(position)/dt = velocity (only include relationships you want enforced exactly)"
                                 if pcnn_mode else "For example: d(position)/dt = velocity; d(velocity)/dt = input / mass"),
                    key=f"physics_source_text_{key}", height=110,
                )
                submit_source = st.form_submit_button(
                    "Check hard-constraint rules" if pcnn_mode else "Check equation source", type="primary",
                )
            st.caption("Text source limit · 120 KB · Python, MATLAB, or plain text")
            if submit_source:
                if uploaded_source:
                    st.session_state["conversation_pending_source_file"] = uploaded_source
                    st.session_state["conversation_pending"] = (
                        "Prepare my attached PCNN hard-constraint source" if pcnn_mode else
                        "Prepare my attached equation source"
                    )
                elif equation_text.strip():
                    st.session_state["conversation_pending"] = (
                        "Prepare these PCNN hard-constraint rules:\n" if pcnn_mode else
                        "Prepare these PINN equations:\n"
                    ) + equation_text.strip()
                else:
                    st.warning("Attach a source file or enter at least one rule before checking.")
        elif stage == "physics_model_description":
            pcnn_mode = str((chat.get("setup") or {}).get("pending_physics_mode", "pinn")).lower() == "pcnn"
            component_key = f"physics_description_editor_{key}"
            component_state = st.session_state.get(component_key, {})
            old_widget_value = st.session_state.get(f"physics_model_description_{key}")
            component_value = (
                component_state.get("value")
                if isinstance(component_state, dict) else None
            )
            initial_description = next(
                (value for value in (old_widget_value, component_value, chat.get("pinn_model_description"))
                 if isinstance(value, str)),
                "",
            )
            description_key = f"physics_model_description_{key}"
            if description_key not in st.session_state:
                st.session_state[description_key] = initial_description

            with st.form(f"physics_model_description_form_{key}", border=False):
                description = st.text_area(
                    "Describe the exact state rules" if pcnn_mode else "Describe the system and its dynamics",
                    max_chars=4000,
                    placeholder=(
                        "Which state-derivative relationships must hold exactly? Name the states; leave unknown states for the neural network."
                        if pcnn_mode else
                        "What does each state represent? What do the inputs do? Add known units, forces, or physical relationships."
                    ),
                    key=description_key,
                    height=150,
                )
                st.caption(
                    "The dataset helps LabCD map the rules to confirmed signals. Review every proposed hard constraint before enabling it."
                    if pcnn_mode else
                    "The dataset helps LabCD map signals and check the model. Review its derivative equations and assumptions before use."
                )
                submit_description = st.form_submit_button(
                    "Suggest hard constraints" if pcnn_mode else "Draft PINN model for review", type="primary"
                )

            if submit_description:
                cleaned_description = description.strip()
                if len(cleaned_description) < 12:
                    st.warning("Add more detail about the exact state relationship PCNN should enforce." if pcnn_mode else
                               "Add a little more detail about the system before drafting its model.")
                else:
                    st.session_state["conversation_pending"] = (
                        "__labcd_physics_description__:" + cleaned_description
                    )
        elif stage == "physics_model_review":
            pcnn_mode = str((chat.get("setup") or {}).get("pending_physics_mode", "pinn")).lower() == "pcnn"
            st.caption("Review each proposed hard-constraint rule and its state mapping; only approved rules can be selected for exact enforcement." if pcnn_mode else
                       "Review the derivative model and its assumptions, then approve or discard it to continue.")

        if question_timer is not None:
            question_timer()


def render_state_comparison(comparison: dict):
    rows = comparison.get("rows") or []
    if not rows:
        return
    st.caption(f"SAVED RUN SCORES · verification_summary.json · {comparison.get('aligned_samples') or 'unknown'} aligned held-out samples")
    st.caption("Confirmed angles use shortest circular error across ±π; other states use ordinary linear error.")
    ranked = [r for r in rows if r["normalized_rmse"] is not None]
    if ranked:
        best = ranked[0]
        st.markdown(f"**Lowest normalized error: `{best['state']}`** · "
                    f"{100 * best['normalized_rmse']:.2f}% of held-out variation")
    table_rows = []
    rank = 0
    for row in rows:
        if row["normalized_rmse"] is not None:
            rank += 1
            rank_label = str(rank)
            normalized = f"{100 * row['normalized_rmse']:.2f}%"
        else:
            rank_label, normalized = "—", "—"
        table_rows.append({"Rank": rank_label, "State": row["state"],
                           "Error comparison": "Circular angle" if row.get("error_type") == "circular" else "Linear state",
                           "Normalized RMSE · lower is better": normalized,
                           "RMSE": f"{row['rmse']:.4g}", "MAE": f"{row['mae']:.4g}",
                           "Samples": row["valid_samples"]})
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch",
                 height=min(310, 42 + 35 * len(table_rows)))
    st.caption("States are ranked by normalized RMSE on the held-out rollout. Lower means closer predictions relative to that state's variation.")
    trajectory_rows = []
    for row in rows:
        for item in row.get("per_trajectory", []):
            ratio = item.get("normalized_rmse")
            trajectory_rows.append({
                "Held-out trajectory": item.get("trajectory_index"),
                "State": row["state"],
                "Error comparison": "Circular angle" if row.get("error_type") == "circular" else "Linear state",
                "Normalized RMSE": f"{100 * float(ratio):.2f}%" if ratio is not None and _finite(ratio) else "Not available",
                "RMSE": f"{float(item['rmse']):.4g}" if item.get("rmse") is not None and _finite(item["rmse"]) else "—",
                "Samples": item.get("valid_samples", 0),
            })
    if trajectory_rows:
        with st.expander("Compare each held-out trajectory"):
            st.dataframe(pd.DataFrame(trajectory_rows), hide_index=True, width="stretch")
            st.caption("Each trajectory is rolled out independently. The ranked score above pools their aligned samples; this table shows whether results vary between trajectories.")
    protocol = comparison.get("protocol") or {}
    if protocol.get("initialization"):
        st.caption("Rollout protocol: " + protocol["initialization"])
