"""Neutral workspace styling for LabCD's conversation interface."""

CSS = """
<style>
/* Native widgets retain their keyboard behavior and accessibility names. */
:root { --chat-bg:#212121; --chat-panel:#2b2b2b; --chat-border:#3a3a3a; --chat-muted:#a7a7a7; }
[data-testid="stAppViewContainer"], .stApp { background:var(--chat-bg); }
[data-testid="stHeader"] { background:transparent; height:0; overflow:visible; pointer-events:none; }
[data-testid="stToolbar"] { height:0; overflow:visible; }
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer { display:none!important; }
[data-testid="stSidebarCollapseButton"] { visibility:visible!important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] {
  pointer-events:auto; width:32px; height:32px; border:0; color:#ccc; background:transparent;
}
[data-testid="stExpandSidebarButton"] { position:fixed; left:12px; top:14px; z-index:1001; }
.stMainBlockContainer { max-width:1500px; padding:1.1rem 2.6rem 2rem; }
[data-testid="stSidebar"] { background:#171717; border-right:1px solid #292929; }
[data-testid="stSidebar"][aria-expanded="true"] { min-width:250px; max-width:290px; }
[data-testid="stSidebar"][aria-expanded="false"] { min-width:0!important; max-width:0!important; width:0!important; }
.stApp:has([data-testid="stExpandSidebarButton"]) .st-key-conversation_header { padding-left:25px; }
[data-testid="stSidebarContent"] { padding-top:.3rem; }
[data-testid="stSidebarHeader"] { height:32px; min-height:32px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.45rem; }
[data-testid="stSidebar"] button { text-align:left; justify-content:flex-start; border-color:transparent; background:transparent; }
[data-testid="stSidebar"] button:hover { background:#292929; }
[data-testid="stSidebar"] button p { font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
[class*="st-key-conversation_nav_selected"] button { background:#303030; }
[class*="st-key-conversation_nav"] button p { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:13px; }
.conversation-brand { display:flex; align-items:center; gap:9px; padding:15px 0 24px; font-size:17px; font-weight:600; }
.labcd-symbol { width:25px; height:25px; border:1px solid #777; border-radius:7px; display:grid; place-items:center; font-size:14px; }
.brand-muted { color:#888; font-size:11px; font-weight:400; padding-top:4px; }
.sidebar-footnote { color:#888; font-size:11px; line-height:1.8; padding:35px 8px 15px; }
[data-testid="stLayoutWrapper"]:has(> .st-key-conversation_header) { position:sticky; top:0; z-index:30; background:var(--chat-bg); }
.st-key-conversation_header { padding:10px 0 18px; margin-bottom:12px; }
.conversation-heading { font-size:14px; color:#ddd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.conversation-heading span { color:#727272; margin:0 9px; }
.st-key-conversation_header button { background:transparent; border-color:transparent; font-size:12px; min-height:34px; }
.st-key-conversation_header button:hover { background:#303030; }
.st-key-conversation_header [data-testid="stPopover"] button p { font-size:0; }
.st-key-conversation_header [data-testid="stPopover"] button { padding:6px; }
.st-key-conversation_center { max-width:820px; margin:0 auto; }
.conversation-welcome { text-align:center; padding:clamp(48px,10vh,120px) 0 27px; }
.welcome-glyph { font-size:51px; font-weight:300; color:#e0e0e0; height:58px; margin-bottom:17px; }
.conversation-welcome h1 { font-family:Arial,sans-serif; font-size:32px; font-weight:500; letter-spacing:-1px; margin:0 0 14px; padding:0; color:#ececec; }
.conversation-welcome p { font-size:15px; color:#aaa; line-height:1.8; }
.st-key-conversation_suggestions { margin:3px auto 55px; }
.st-key-conversation_suggestions button { border:1px solid #424242; background:transparent; border-radius:12px; min-height:45px; font-size:12px; }
.st-key-conversation_suggestions button:hover { background:#303030; border-color:#777; }
[data-testid="stChatMessage"] { background:transparent; padding:1.15rem .25rem; border:0; gap:14px; }
[data-testid="stChatMessage"] p { font-size:15px; line-height:1.8; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#303030; border-radius:18px; margin:14px 0 10px auto; padding:16px 20px; max-width:90%; }
[data-testid="stChatMessageAvatarUser"] { display:none; }
[data-testid="stChatMessageAvatarAssistant"] { background:transparent; color:#dadada; }
[data-testid="stChatMessage"] [data-testid="stExpander"] { border-color:#414141; border-radius:10px; }
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] { border-color:#414141; border-radius:12px; }
[data-testid="stBottomBlockContainer"] { background:var(--chat-bg); padding:0 2.6rem .7rem; }
.st-key-conversation_composer { max-width:820px; margin:0 auto; background:var(--chat-bg); padding-top:8px; }
.st-key-conversation_composer [data-testid="stChatInput"] { border:1px solid #4a4a4a; border-radius:21px; background:#303030; min-height:76px; box-shadow:0 4px 22px #0002; }
.st-key-conversation_composer [data-testid="stChatInput"]:focus-within { border-color:#8a8a8a; }
.st-key-conversation_composer textarea { font-size:15px; line-height:1.6; background:transparent; }
.st-key-conversation_composer button { border-color:transparent; background:transparent; min-height:30px; font-size:12px; }
.st-key-conversation_composer [data-testid="stCaptionContainer"] p { font-size:11px; color:#a0a0a0; }
.st-key-conversation_artifacts { background:#262626; border-color:#424242; border-radius:14px; padding:16px; }
.st-key-conversation_artifacts h4 { font-family:Arial,sans-serif; font-size:17px; font-weight:500; }
[class*="st-key-run_setup_card_"] [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3c5350; border-radius:18px;
  background:linear-gradient(145deg,#262c2c 0%,#222426 58%,#24242a 100%);
  box-shadow:0 12px 32px #0002,inset 0 1px #ffffff08;
}
.architecture-diagram { display:block; width:100%; height:auto; margin:1px 0 8px; overflow:visible; }
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
.pinn-guidance-diagram { display:block; width:100%; height:auto; margin:8px 0; }
.pinn-panel { fill:#111d20; stroke:#38504c; stroke-width:1.2; }
.pinn-panel.model { fill:#182a29; stroke:#58aa98; }
.pinn-panel.output { fill:#1d2230; stroke:#56699f; }
.pinn-heading { fill:#d9e9e4; font:600 10px ui-monospace,monospace; letter-spacing:.6px; }
.pinn-sub { fill:#94aaa4; font:10px Arial,sans-serif; }
.pinn-grid { fill:none; stroke:#35504d; stroke-width:1; }
.pinn-wave { fill:none; stroke:#70dcc4; stroke-width:2.4; stroke-linecap:round; }
.pinn-link,.pinn-equation-link { fill:none; stroke:#7dd1bd; stroke-width:1.8; stroke-dasharray:5 6; animation:pinn-flow 1.6s linear infinite; }
.pinn-equation-link { stroke:#a6b3ff; }
.pinn-arrow { fill:none; stroke:#83cfbc; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.pinn-neuron { fill:#253e3a; stroke:#70b7a8; stroke-width:1.3; }
.pinn-neuron.bright { fill:#70dcc4; stroke:#bbfff0; animation:arch-neuron-pulse 1.9s ease-in-out infinite; }
.pinn-equation { fill:#21283a; stroke:#7584c4; stroke-width:1.1; }
.pinn-equation-text { fill:#c5ceff; font:600 10px ui-monospace,monospace; }
@keyframes pinn-flow { to { stroke-dashoffset:-22; } }
@media(max-width:620px) { .pinn-guidance-diagram { min-width:650px; } }
.architecture-diagram.unselected { opacity:.76; transition:opacity .2s ease; }
.architecture-diagram.selected { opacity:1; }
.arch-choice-selected, .model-choice-selected { display:inline-flex; align-items:center; gap:6px; border:1px solid #4caa98;
  border-radius:999px; padding:4px 9px; margin:0 0 7px; color:#b9f6e9; background:#183d36;
  font:600 9px/1.4 ui-monospace,monospace; letter-spacing:.08em; box-shadow:0 0 14px #4ec8ad28; }
[class*="st-key-setup_model_card_lstm_"]:has(.model-choice-selected) [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-setup_model_card_mlp_"]:has(.model-choice-selected) [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#55c7b1; background:linear-gradient(145deg,#20332f 0%,#242829 64%,#252730 100%);
  box-shadow:0 0 0 1px #55c7b144,0 0 26px #42c4a32b,inset 0 1px #d8fff31a;
  animation:model-choice-glow 2.8s ease-in-out infinite;
}
[class*="st-key-setup_model_card_lstm_"]:not(:has(.model-choice-selected)) .architecture-diagram,
[class*="st-key-setup_model_card_mlp_"]:not(:has(.model-choice-selected)) .architecture-diagram { opacity:.63; }
@keyframes model-choice-glow { 0%,100% { box-shadow:0 0 0 1px #55c7b133,0 0 18px #42c4a31c,inset 0 1px #d8fff310; }
  50% { box-shadow:0 0 0 1px #65e2cb77,0 0 30px #42c4a34a,inset 0 1px #d8fff322; } }
@keyframes arch-neuron-pulse { 0%,100% { filter:drop-shadow(0 0 1px #70e3c455); } 50% { filter:drop-shadow(0 0 5px #70e3c4bb); } }
@keyframes arch-signal-flow { to { stroke-dashoffset:-18; } }
[class*="st-key-setup_model_card_lstm_"] [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-setup_model_card_mlp_"] [data-testid="stVerticalBlockBorderWrapper"] { transition:border-color .2s ease,background .2s ease,box-shadow .2s ease; }
[class*="st-key-setup_pinn_panel_"] [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3b4c4b; border-radius:14px;
  background:linear-gradient(120deg,#1d2929 0%,#202326 65%,#24242a 100%);
}
[class*="st-key-setup_pinn_panel_"] [data-testid="stCaptionContainer"] p {
  color:#9badaa; font-size:12px; line-height:1.55;
}
.st-key-conversation_file_panel {
  position:fixed; top:76px; right:24px; z-index:25;
  width:min(28vw,360px); max-height:calc(100dvh - 100px); box-sizing:border-box;
}
.st-key-conversation_file_panel [data-testid="stVerticalBlockBorderWrapper"] {
  max-height:calc(100dvh - 100px); overflow-y:auto;
  border-color:#3f3f3f; background:#202020; border-radius:14px;
}
[class*="st-key-chat_file_card"] [data-testid="stVerticalBlockBorderWrapper"] { border-color:#393939; background:#282828; border-radius:12px; }
[class*="st-key-chat_file_card"] button { background:transparent; border:0; text-align:left; justify-content:flex-start; min-height:34px; }
[class*="st-key-chat_file_card"] button:hover { background:#333; }
[class*="st-key-chat_file_card"] [data-testid="stCaptionContainer"] p { font-size:11px; }
[class*="st-key-chat_file_card_zip_"] [data-testid="stVerticalBlockBorderWrapper"] {
  min-height:132px; border-color:#67736f;
  background:linear-gradient(110deg,#2d3533 0%,#292b2c 58%,#2b2a2b 100%);
  border-radius:15px; box-shadow:0 8px 24px #0003,inset 0 1px #ffffff0a;
}
[class*="st-key-chat_file_card_zip_"] button {
  min-height:62px; padding:10px 16px; border:1px solid #505957;
  border-radius:11px; background:#303735; transition:background .16s ease,border-color .16s ease;
}
[class*="st-key-chat_file_card_zip_"] button:hover { background:#39423f; border-color:#84908b; }
[class*="st-key-chat_file_card_zip_"] button p { font-size:16px; font-weight:600; }
[class*="st-key-chat_file_card_zip_"] [data-testid="stCaptionContainer"] p { font-size:12px; }
.st-key-conversation_live_run_card [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3d514d; border-radius:18px;
  background:radial-gradient(ellipse at 8% 0%,#263b37 0%,#242626 42%,#222222 100%);
  box-shadow:0 12px 34px #0003,inset 0 1px #ffffff08;
}
.st-key-conversation_live_run_card [data-testid="stMarkdownContainer"] h3 {
  margin:.4rem 0 .2rem; font-size:1.2rem; letter-spacing:-.025em; color:#eef4f1;
}
.run-live-label { display:flex; align-items:center; gap:9px; color:#cce9df; font-size:13px; font-weight:650; }
.run-live-dot { width:9px; height:9px; border-radius:50%; background:#68d6ad; box-shadow:0 0 0 4px #68d6ad20; animation:run-live-pulse 1.8s ease-in-out infinite; }
@keyframes run-live-pulse { 0%,100% { box-shadow:0 0 0 3px #68d6ad18; opacity:1; } 50% { box-shadow:0 0 0 7px #68d6ad08; opacity:.72; } }
.st-key-conversation_live_run_card [data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#69c5a7,#91d8bd); }
.run-stepper { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:3px 0 12px; }
.run-step { display:flex; align-items:center; gap:7px; min-width:0; padding:8px 9px; border:1px solid #383d3b; border-radius:10px; color:#8d9290; background:#202221a8; font-size:11px; transition:background .2s ease,border-color .2s ease,color .2s ease; }
.run-step > span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.run-step-dot { display:grid; flex:0 0 19px; width:19px; height:19px; place-items:center; border-radius:50%; background:#343836; color:#aaa; font-size:10px; font-weight:700; }
.run-step.done { border-color:#36554b; color:#c3ded3; background:#23322d; }
.run-step.done .run-step-dot { color:#17261f; background:#81cbae; }
.run-step.active { border-color:#54796e; color:#e2f1eb; background:linear-gradient(110deg,#29433b,#2b302e); box-shadow:inset 0 0 0 1px #6ab99c20; }
.run-step.active .run-step-dot { color:#18332a; background:#8bd6b5; box-shadow:0 0 0 4px #8bd6b51c; }
.st-key-conversation_live_run_card [data-testid="stMetric"] { min-height:76px; padding:10px 12px; border:1px solid #383d3b; border-radius:11px; background:#202221a8; }
.st-key-conversation_live_run_card [data-testid="stMetricLabel"] p { color:#a4aaa7; font-size:11px; }
.st-key-conversation_live_run_card [data-testid="stMetricValue"] { color:#e4ede9; font-size:1.05rem; }
.st-key-conversation_live_run_card [data-testid="stExpander"] { border-color:#414744; border-radius:11px; background:#20222180; }
.st-key-conversation_live_run_card [data-testid="stCaptionContainer"] p { line-height:1.55; }
[data-testid="stCaptionContainer"] { color:var(--chat-muted); }
button:focus-visible { outline:2px solid #b9b9b9; outline-offset:2px; }
.st-key-conversation_composer textarea:focus-visible { outline:none; }
@media(max-width:850px) {
  .stMainBlockContainer { padding:1rem 1rem 2rem; }
  [data-testid="stBottomBlockContainer"] { padding:0 1rem .5rem; }
  .conversation-welcome { padding-top:8vh; }
  .conversation-welcome h1 { font-size:27px; }
  .conversation-heading { font-size:12px; }
  .run-stepper { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .st-key-conversation_file_panel {
    top:auto; right:12px; bottom:102px; left:12px; width:auto;
    max-height:55dvh;
  }
  .st-key-conversation_file_panel [data-testid="stVerticalBlockBorderWrapper"] { max-height:55dvh; }
}
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto!important; transition:none!important; animation:none!important; } }
</style>
"""
