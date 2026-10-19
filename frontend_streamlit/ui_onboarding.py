"""A short, visual first-run welcome and guided tour for the LabCD workspace."""

from __future__ import annotations

import streamlit as st

try:
    from . import ui_brand
except ImportError:
    import ui_brand


_WELCOME_SCENE = r"""
<svg class="labcd-tour-visual" viewBox="0 0 960 230" role="img" aria-label="Measurements flow through a model and become predictions">
  <rect class="labcd-svg-card" x="18" y="34" width="260" height="162" rx="15"/>
  <text class="labcd-svg-label" x="38" y="60">YOUR MEASUREMENTS</text>
  <text class="labcd-svg-muted" x="38" y="78">time · s_* states · a_* inputs</text>
  <text class="labcd-svg-muted" x="38" y="95">one xdot_* rate for each state</text>
  <path class="labcd-svg-grid" d="M39 103H257M39 132H257M39 161H257M82 94V177M130 94V177M178 94V177M226 94V177"/>
  <path class="labcd-svg-line" d="M40 153 C57 151 59 111 78 114 S99 166 118 157 S139 105 158 116 S181 149 199 135 S226 105 257 108"/>
  <path class="labcd-svg-dash" d="M279 115H366"/>
  <rect class="labcd-svg-teal" x="366" y="34" width="232" height="162" rx="17"/>
  <text class="labcd-svg-label" x="482" y="60" text-anchor="middle">LEARN THE DYNAMICS</text>
  <text class="labcd-svg-muted" x="482" y="78" text-anchor="middle">model · test · improve</text>
  <path class="labcd-svg-grid" d="M394 107L433 129L394 157M433 129L482 100M433 129L482 158M482 100L528 129M482 158L528 129M528 129L570 107M528 129L570 157"/>
  <circle class="labcd-svg-neuron" cx="394" cy="107" r="7"/><circle class="labcd-svg-neuron" cx="394" cy="157" r="7"/>
  <circle class="labcd-svg-neuron" cx="433" cy="129" r="8"/><circle class="labcd-svg-neuron" cx="482" cy="100" r="7"/><circle class="labcd-svg-neuron" cx="482" cy="158" r="7"/>
  <circle class="labcd-svg-neuron" cx="528" cy="129" r="8"/><circle class="labcd-svg-neuron" cx="570" cy="107" r="7"/><circle class="labcd-svg-neuron" cx="570" cy="157" r="7"/>
  <path class="labcd-svg-dash" d="M598 115H683"/>
  <rect class="labcd-svg-blue" x="683" y="34" width="258" height="162" rx="15"/>
  <text class="labcd-svg-label" x="703" y="60">PREDICTED BEHAVIOR</text>
  <text class="labcd-svg-muted" x="703" y="78">compare it with the measurements</text>
  <path class="labcd-svg-grid" d="M704 108H920M704 137H920M704 166H920"/>
  <path class="labcd-svg-line" d="M704 158 C727 157 731 119 755 121 S783 165 806 155 S832 115 855 121 S884 150 920 112"/>
  <path class="labcd-svg-forecast" d="M855 121 C878 127 888 137 920 109"/>
  <circle class="labcd-svg-dot" cx="482" cy="100" r="5"/>
</svg>
"""


_SCENES = (
    r"""<svg class="labcd-tour-visual" viewBox="0 0 960 230" role="img" aria-label="Attach a spreadsheet and preview its measured signal">
      <rect class="labcd-svg-card" x="24" y="27" width="250" height="176" rx="16"/>
      <path class="labcd-svg-blue" d="M103 54h75l30 30v76H103z"/><path class="labcd-svg-grid" d="M178 54v31h30M122 104h68M122 122h68M122 140h47"/>
      <rect class="labcd-svg-teal" x="46" y="154" width="93" height="28" rx="14"/><text class="labcd-svg-muted" x="92" y="172" text-anchor="middle">CSV · XLSX</text>
      <text class="labcd-svg-label" x="242" y="77">DROP OR PASTE</text><path class="labcd-svg-dash" d="M274 115H374"/>
      <rect class="labcd-svg-card" x="374" y="27" width="562" height="176" rx="16"/>
      <text class="labcd-svg-label" x="398" y="56">A QUICK LOOK AT YOUR SIGNAL</text><text class="labcd-svg-muted" x="398" y="75">time on the horizontal axis · measured state on the vertical</text>
      <path class="labcd-svg-grid" d="M401 105H910M401 137H910M401 169H910M451 94V183M527 94V183M603 94V183M679 94V183M755 94V183M831 94V183M907 94V183"/>
      <path class="labcd-svg-line" d="M402 164 C436 164 436 145 470 145 S508 112 542 118 S584 159 616 152 S654 103 690 111 S730 151 764 138 S811 107 842 116 S879 147 909 103"/>
      <circle class="labcd-svg-dot" cx="690" cy="111" r="5"/><text class="labcd-svg-muted" x="402" y="196">Required to train: time, measured state(s), and input(s).</text>
    </svg>""",
    r"""<svg class="labcd-tour-visual" viewBox="0 0 960 230" role="img" aria-label="Confirm whether measurements are continuous or stacked and identify angle states">
      <rect class="labcd-svg-card" x="18" y="27" width="444" height="176" rx="16"/><rect class="labcd-svg-card" x="498" y="27" width="444" height="176" rx="16"/>
      <text class="labcd-svg-label" x="40" y="55">ONE CONTINUOUS EXPERIMENT</text><text class="labcd-svg-label" x="520" y="55">SEVERAL RUNS IN ONE FILE</text>
      <path class="labcd-svg-grid" d="M43 91H437M43 124H437M43 157H437M91 80V174M169 80V174M247 80V174M325 80V174M403 80V174M523 91H917M523 124H917M523 157H917M571 80V174M649 80V174M727 80V174M805 80V174M883 80V174"/>
      <path class="labcd-svg-line" d="M44 154 C85 151 94 140 129 138 S188 121 223 117 S284 106 318 108 S382 87 436 90"/>
      <path class="labcd-svg-line-blue" d="M524 154 C561 148 587 130 620 119 S674 86 710 90 L728 153 M729 153 C757 148 774 125 810 116 S853 93 880 98 L900 153 M901 153L917 142"/>
      <path class="labcd-svg-forecast" d="M728 78V177M900 78V177"/><text class="labcd-svg-muted" x="40" y="191">No restart between samples</text><text class="labcd-svg-muted" x="520" y="191">A restart begins another trajectory</text>
      <rect class="labcd-svg-teal" x="386" y="8" width="188" height="31" rx="15"/><text class="labcd-svg-muted" x="480" y="28" text-anchor="middle">time · state · input · angle?</text>
    </svg>""",
    r"""<svg class="labcd-tour-visual" viewBox="0 0 960 230" role="img" aria-label="Review a suggested model and search effort before training">
      <rect class="labcd-svg-card" x="18" y="54" width="193" height="132" rx="15"/><text class="labcd-svg-label" x="114" y="81" text-anchor="middle">STATE + INPUT</text>
      <rect class="labcd-svg-blue" x="47" y="98" width="135" height="28" rx="8"/><text class="labcd-svg-muted" x="114" y="116" text-anchor="middle">recent measurements</text>
      <rect class="labcd-svg-teal" x="47" y="137" width="135" height="28" rx="8"/><text class="labcd-svg-muted" x="114" y="155" text-anchor="middle">control signals</text>
      <path class="labcd-svg-dash" d="M211 119H314"/><rect class="labcd-svg-teal" x="314" y="29" width="339" height="180" rx="17"/><text class="labcd-svg-label" x="483" y="54" text-anchor="middle">INITIALIZER'S STARTING MODEL</text>
      <path class="labcd-svg-grid" d="M353 93L406 119L353 151M406 119L460 88M406 119L460 151M460 88L511 119M460 151L511 119M511 119L565 91M511 119L565 150M565 91L614 119M565 150L614 119"/>
      <circle class="labcd-svg-neuron" cx="353" cy="93" r="8"/><circle class="labcd-svg-neuron" cx="353" cy="151" r="8"/><circle class="labcd-svg-neuron" cx="406" cy="119" r="9"/><circle class="labcd-svg-neuron" cx="460" cy="88" r="8"/><circle class="labcd-svg-neuron" cx="460" cy="151" r="8"/><circle class="labcd-svg-neuron" cx="511" cy="119" r="9"/><circle class="labcd-svg-neuron" cx="565" cy="91" r="8"/><circle class="labcd-svg-neuron" cx="565" cy="150" r="8"/><circle class="labcd-svg-neuron" cx="614" cy="119" r="9"/>
      <circle class="labcd-svg-dot" cx="460" cy="88" r="5"/><path class="labcd-svg-dash" d="M653 119H724"/>
      <rect class="labcd-svg-card" x="724" y="54" width="218" height="132" rx="15"/><text class="labcd-svg-label" x="833" y="81" text-anchor="middle">YOUR REVIEW</text>
      <rect class="labcd-svg-blue" x="750" y="98" width="166" height="27" rx="8"/><text class="labcd-svg-muted" x="833" y="116" text-anchor="middle">LSTM or MLP</text>
      <rect class="labcd-svg-purple" x="750" y="137" width="166" height="27" rx="8"/><text class="labcd-svg-muted" x="833" y="155" text-anchor="middle">fast · regular · heavy</text>
    </svg>""",
    r"""<svg class="labcd-tour-visual" viewBox="0 0 960 230" role="img" aria-label="Inspect held-out predictions, cycle scores, and run files">
      <rect class="labcd-svg-card" x="18" y="26" width="581" height="178" rx="16"/><text class="labcd-svg-label" x="42" y="54">MEASURED VS MODEL PREDICTION</text><text class="labcd-svg-muted" x="42" y="72">held-out data shows how well the dynamics generalize</text>
      <path class="labcd-svg-grid" d="M45 107H572M45 139H572M45 171H572M101 91V186M183 91V186M265 91V186M347 91V186M429 91V186M511 91V186"/>
      <path class="labcd-svg-line" d="M47 163 C82 161 94 124 130 127 S177 169 214 157 S258 110 293 116 S343 156 377 145 S421 105 458 112 S515 152 570 101"/>
      <path class="labcd-svg-forecast" d="M47 167 C83 159 96 128 130 130 S177 164 214 154 S258 115 293 119 S343 152 377 141 S421 109 458 116 S515 146 570 106"/>
      <rect class="labcd-svg-card" x="620" y="26" width="322" height="178" rx="16"/><text class="labcd-svg-label" x="645" y="54">SEARCH CYCLES</text><text class="labcd-svg-muted" x="645" y="72">compare candidates, then ask what changed</text>
      <path class="labcd-svg-grid" d="M650 178H912"/><rect class="labcd-svg-teal" x="663" y="133" width="40" height="45" rx="5"/><rect class="labcd-svg-blue" x="718" y="111" width="40" height="67" rx="5"/><rect class="labcd-svg-teal" x="773" y="91" width="40" height="87" rx="5"/><rect class="labcd-svg-blue" x="828" y="104" width="40" height="74" rx="5"/><rect class="labcd-svg-purple" x="883" y="78" width="40" height="100" rx="5"/>
      <circle class="labcd-svg-dot" cx="903" cy="78" r="5"/><text class="labcd-svg-muted" x="663" y="194">1</text><text class="labcd-svg-muted" x="718" y="194">2</text><text class="labcd-svg-muted" x="773" y="194">3</text><text class="labcd-svg-muted" x="828" y="194">4</text><text class="labcd-svg-muted" x="883" y="194">5</text>
    </svg>""",
)


_TOUR_STEPS = (
    {
        "title": "Add measurements",
        "body": "Attach a CSV or Excel file, or paste a table into chat. Time plus one measured state is enough to begin.",
        "tip": "Unclear headers are fine. Tell LabCD what the columns mean and it can help map them.",
    },
    {
        "title": "Confirm what the data means",
        "body": "When needed, LabCD asks whether the data is one run or several, which columns are states and inputs, and whether an angle wraps.",
        "tip": "You can correct a choice or say you are unsure; LabCD can guide you gently.",
    },
    {
        "title": "Review the model and search",
        "body": "The initializer suggests LSTM or MLP settings. Review the recommendation, choose search effort, and set the number of cycles.",
        "tip": "You stay in control: change the suggestion before approving the run.",
    },
    {
        "title": "Run, inspect, and ask",
        "body": "Follow the cycles, inspect held-out plots and files, then ask what worked, what failed, or what experiment to try next.",
        "tip": "Ask what happened in this run; LabCD can help you decide the next step.",
    },
)


@st.dialog("Welcome to LabCD", width="large", dismissible=False, icon=":material/insights:")
def _onboarding_dialog():
    mode = st.session_state.get("labcd_onboarding_mode", "welcome")
    if mode == "welcome":
        st.markdown(
            f'<div class="labcd-intro-kicker"><span class="labcd-onboarding-mark">{ui_brand.ICON_MARKUP}</span>'
            '<span>A visual quick start</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="labcd-intro-title">From measurements to a model, through chat.</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="labcd-intro-copy">System identification means learning how a real system behaves from measurements. '
            'LabCD helps organize signals, train and test a model, and explore results with you.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(_WELCOME_SCENE, unsafe_allow_html=True)
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button("Close", icon=":material/close:", key="labcd_intro_close"):
                st.session_state["labcd_onboarding_seen"] = True
                st.rerun()
            if st.button("Take the tutorial", icon=":material/school:", type="primary", key="labcd_intro_tutorial"):
                st.session_state["labcd_onboarding_mode"] = "tutorial"
                st.session_state["labcd_onboarding_step"] = 0
                st.rerun()
        return

    step_index = min(max(int(st.session_state.get("labcd_onboarding_step", 0)), 0), len(_TOUR_STEPS) - 1)
    step = _TOUR_STEPS[step_index]
    progress = "".join(
        f'<span class="{"done" if index <= step_index else ""}"></span>'
        for index in range(len(_TOUR_STEPS))
    )
    st.markdown(f'<div class="labcd-tour-progress" aria-label="Tutorial progress">{progress}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="labcd-tour-count">STEP {step_index + 1:02d} / {len(_TOUR_STEPS):02d}</div>', unsafe_allow_html=True)
    st.markdown(_SCENES[step_index], unsafe_allow_html=True)
    st.markdown(f'<div class="labcd-intro-title">{step["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="labcd-intro-copy">{step["body"]}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="labcd-tour-tip"><span>✦</span><span>{step["tip"]}</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, horizontal_alignment="distribute", gap="small"):
        if st.button("Back", icon=":material/arrow_back:", key="labcd_tour_back", disabled=step_index == 0):
            st.session_state["labcd_onboarding_step"] = step_index - 1
            st.rerun()
        if st.button("Close tutorial", icon=":material/close:", key="labcd_tour_close"):
            st.session_state["labcd_onboarding_seen"] = True
            st.rerun()
        is_last = step_index == len(_TOUR_STEPS) - 1
        if st.button("Finish" if is_last else "Next", icon=":material/check:" if is_last else ":material/arrow_forward:",
                     type="primary", key="labcd_tour_next"):
            if is_last:
                st.session_state["labcd_onboarding_seen"] = True
            else:
                st.session_state["labcd_onboarding_step"] = step_index + 1
            st.rerun()


def render_onboarding() -> None:
    """Show the first-visit modal or reopen the guided tutorial on request."""
    tutorial_requested = bool(st.session_state.pop("labcd_tutorial_request", False))
    if tutorial_requested:
        st.session_state["labcd_onboarding_mode"] = "tutorial"
        st.session_state["labcd_onboarding_step"] = 0
    elif st.session_state.get("labcd_onboarding_seen", False):
        return
    else:
        st.session_state.setdefault("labcd_onboarding_mode", "welcome")
    _onboarding_dialog()
