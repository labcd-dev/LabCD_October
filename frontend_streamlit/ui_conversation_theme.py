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
.conversation-brand { display:flex; align-items:center; gap:0; padding:15px 0 24px; font-size:17px; font-weight:600; }
.conversation-brand .labcd-logo { width:116px; height:61px; flex:none; mix-blend-mode:lighten; }
.labcd-icon { display:block; width:40px; height:40px; mix-blend-mode:lighten; }
.brand-muted { color:#888; font-size:11px; font-weight:400; padding-top:4px; }
.sidebar-footnote { color:#888; font-size:11px; line-height:1.8; padding:35px 8px 15px; }
[data-testid="stLayoutWrapper"]:has(> .st-key-conversation_header) { position:sticky; top:0; z-index:30; background:var(--chat-bg); }
.st-key-conversation_header { padding:10px 0 18px; margin-bottom:12px; }
.conversation-heading { display:flex; align-items:center; gap:10px; min-width:0; font-size:14px; color:#ddd; overflow:hidden; white-space:nowrap; }
.conversation-heading .header-brand { display:inline-flex; flex:none; align-items:center; }
.conversation-heading .labcd-logo { width:116px; height:61px; mix-blend-mode:lighten; }
.conversation-heading .conversation-heading-separator { color:#727272; flex:none; }
.conversation-heading-title { overflow:hidden; text-overflow:ellipsis; }
.st-key-conversation_header button { background:transparent; border-color:transparent; font-size:12px; min-height:34px; }
.st-key-conversation_header button:hover { background:#303030; }
.st-key-conversation_header [data-testid="stPopover"] button p { font-size:0; }
.st-key-conversation_header [data-testid="stPopover"] button { padding:6px; }
.st-key-conversation_center { max-width:820px; margin:0 auto; }
.conversation-welcome { text-align:center; padding:clamp(48px,10vh,120px) 0 27px; }
.welcome-glyph { display:flex; justify-content:center; height:58px; margin-bottom:17px; }
.welcome-glyph .labcd-icon { width:56px; height:56px; filter:drop-shadow(0 5px 18px #756cff44); mix-blend-mode:lighten; }
.conversation-welcome h1 { font-family:Arial,sans-serif; font-size:32px; font-weight:500; letter-spacing:-1px; margin:0 0 14px; padding:0; color:#ececec; }
.conversation-welcome p { font-size:15px; color:#aaa; line-height:1.8; }
[class*="st-key-excel_help_sheet_"] { max-width:640px; margin:16px 0 10px; text-align:left; }
[class*="st-key-excel_help_sheet_"] [data-testid="stVerticalBlockBorderWrapper"] {
  padding:8px 12px; border-color:#34353d; border-radius:15px; background:#141417;
  box-shadow:inset 0 1px #ffffff08,0 8px 22px #0002;
}
[class*="st-key-excel_help_sheet_"] [data-testid="stMarkdownContainer"] p { margin-bottom:0; color:#e3e3e8; font-size:13px; }
[class*="st-key-excel_help_sheet_"] [data-testid="stCaptionContainer"] p { color:#92929c; font-size:10px; line-height:1.5; }
[class*="st-key-excel_help_sheet_"] [data-testid="stDataFrame"] { border:1px solid #292a31; border-radius:10px; overflow:hidden; }
.st-key-conversation_suggestions { margin:3px auto 55px; }
.st-key-conversation_suggestions button { border:1px solid #424242; background:transparent; border-radius:12px; min-height:45px; font-size:12px; }
.st-key-conversation_suggestions button:hover { background:#303030; border-color:#777; }
[data-testid="stChatMessage"] { background:transparent; padding:1.15rem .25rem; border:0; gap:14px; }
[data-testid="stChatMessage"] p { font-size:15px; line-height:1.8; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
  box-sizing:border-box; width:min(90%,1080px); max-width:90%;
  justify-self:end; align-self:flex-end; margin:14px 0 10px auto!important;
  padding:14px 16px; background:transparent; border:0;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) > [data-testid="stChatMessageContent"] {
  box-sizing:border-box; width:fit-content; max-width:100%; min-width:min(100%,128px); margin-left:auto;
  padding:16px 20px; background:linear-gradient(145deg,#182a40 0%,#172538 100%);
  border:1px solid #477bb8; border-radius:18px 18px 5px 18px;
  box-shadow:inset 0 1px #ffffff0c,0 5px 20px #0003;
  overflow-wrap:anywhere;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) p { color:#edf5ff; }
/* The live-run dashboard is rendered inside an assistant chat row. The global
   custom-avatar rule can otherwise turn the whole dashboard into a blue chat
   bubble whenever Streamlit reruns the page (for example, after a greeting). */
[data-testid="stChatMessage"]:has(.st-key-conversation_live_run_card) {
  box-sizing:border-box; width:100%!important; max-width:100%!important;
  justify-self:stretch!important; align-self:stretch!important;
  margin:0!important; padding:0!important;
  background:transparent!important; border:0!important;
}
[data-testid="stChatMessage"]:has(.st-key-conversation_live_run_card) > [data-testid="stChatMessageContent"] {
  box-sizing:border-box; width:100%!important; max-width:100%!important; min-width:0!important;
  margin:0!important; padding:0!important;
  background:transparent!important; border:0!important; border-radius:0!important;
  box-shadow:none!important;
}
/* Run-chat messages are a separate, lightweight conversation inside the live
   run card. Keep their user bubbles neutral instead of applying the blue
   styling used by the main conversation composer. */
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
  box-sizing:border-box; width:100%!important; max-width:100%!important;
  justify-self:stretch!important; align-self:stretch!important;
  margin:4px 0!important; padding:2px 0!important;
  background:transparent!important; border:0!important;
}
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > [data-testid="stChatMessageContent"],
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) > [data-testid="stChatMessageContent"] {
  width:fit-content!important; max-width:100%!important; min-width:0!important;
  margin:0 auto 0 0!important; padding:9px 12px!important;
  background:#2b2b2b!important; border:0!important; border-radius:12px!important;
  box-shadow:none!important; overflow-wrap:anywhere;
}
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[class*="st-key-talk_agent_messages_"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) p {
  color:#e1e1e1!important;
}
/* Explicit Material avatar strings use Streamlit's Custom test id, not AvatarUser. */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarCustom"] { display:none; }
[data-testid="stChatMessageAvatarAssistant"] { background:transparent; color:#dadada; }
[data-testid="stChatMessageAvatarAssistant"] img { mix-blend-mode:lighten; }
.detection-table-card {
  margin:12px 0 15px; padding:15px; border:1px solid #344944; border-radius:15px;
  background:linear-gradient(145deg,#1a2524 0%,#1c2025 72%,#202129 100%);
  box-shadow:inset 0 1px #ffffff08,0 8px 24px #0002;
}
.detection-table-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }
.detection-table-kicker { color:#79cdb8; font:600 9px/1.4 ui-monospace,monospace; letter-spacing:.15em; }
.detection-table-title { margin-top:3px; color:#e3efeb; font-size:14px; font-weight:700; }
.detection-table-summary { margin-top:3px; color:#9eafaa; font-size:11px; line-height:1.5; }
.detection-table-count { display:inline-flex; flex:none; align-items:center; gap:6px; border:1px solid #3d695f; border-radius:999px; padding:4px 9px; color:#a9eadb; background:#18322d; font:600 9px/1.4 ui-monospace,monospace; }
.detection-table-count::before { content:""; width:6px; height:6px; border-radius:50%; background:#62d1b5; box-shadow:0 0 8px #62d1b588; }
.detection-table-scroll { max-width:100%; overflow-x:auto; border:1px solid #303d3a; border-radius:10px; }
.detection-table { width:100%; min-width:440px; border-collapse:collapse; background:#151b1d; }
.detection-table th { padding:9px 11px; color:#91b4a9; background:#202a2b; font:600 9px/1.4 ui-monospace,monospace; letter-spacing:.06em; text-align:left; white-space:nowrap; }
.detection-table td { padding:9px 11px; color:#d9e5e1; border-top:1px solid #293431; font-size:11px; line-height:1.45; }
.detection-table tbody tr:nth-child(even) td { background:#192021; }
.detection-table tbody tr:hover td { background:#202b29; }
.detection-table-note { margin-top:9px; color:#91a39d; font-size:10px; line-height:1.5; }
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
.sandbox-preview-note {
  display:grid; gap:5px; margin:2px 0 14px; padding:12px 14px;
  border:1px solid #36564f; border-radius:13px;
  background:linear-gradient(115deg,#172a27 0%,#1d2528 68%,#252531 100%);
  box-shadow:inset 0 1px #ffffff0a,0 8px 24px #0002;
}
.sandbox-preview-note span { color:#7ed4bd; font:600 9px/1.4 ui-monospace,monospace; letter-spacing:.13em; }
.sandbox-preview-note b { color:#e5f1ed; font-size:13px; }
.sandbox-preview-note small { color:#9eaca9; font-size:10px; line-height:1.5; }
.st-key-model_sandbox_summary { margin-bottom:10px; }
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
.setup-edit-summary {
  display:flex; align-items:center; gap:12px; margin:3px 0 16px; padding:12px 14px;
  border:1px solid #3c625a; border-radius:13px;
  background:linear-gradient(110deg,#172b28 0%,#1d2428 66%,#22232a 100%);
  box-shadow:inset 0 1px #ffffff08,0 8px 22px #0002;
}
.setup-edit-summary-mark {
  display:grid; place-items:center; width:34px; height:34px; flex:none; border-radius:11px;
  color:#a9f4e1; background:#1b4038; font-size:13px; text-shadow:0 0 12px #71e1c6;
}
.setup-edit-summary strong { display:block; color:#edf7f3; font-size:13px; }
.setup-edit-summary small { display:block; margin-top:2px; color:#a9bbb5; font-size:11px; }
[class*="st-key-run_setup_card_"] [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#385d54; border-radius:17px;
  background:linear-gradient(145deg,#1a2725,#202323 70%);
}
[class*="st-key-setup_chosen_model_"] {
  border:1px solid #5a9b82; border-radius:16px; padding:12px 14px;
  background:radial-gradient(ellipse at 0% 0%,#24463c 0%,#1b2d29 46%,#1b2024 100%);
  box-shadow:0 0 0 1px #64d6ad20,inset 0 1px #ffffff10,0 12px 30px #0003;
}
[class*="st-key-setup_chosen_model_"] h2 {
  margin:.12rem 0 .05rem; color:#f0fff8; font-size:2rem; letter-spacing:-.045em;
  text-shadow:0 0 22px #5fd5ac55;
}
[class*="st-key-setup_chosen_model_"] [data-testid="stMarkdownContainer"] p { margin:.25rem 0; }
[class*="st-key-setup_chosen_model_"] button {
  border:1px solid #5b9e87; border-radius:10px; color:#d8fff0; background:#234237;
}
[class*="st-key-setup_chosen_model_"] button:hover { border-color:#92e8c8; background:#2b5143; }
.setup-choice-kicker { color:#8be4c5; font:700 10px/1.4 ui-monospace,monospace; letter-spacing:.14em; }
.setup-choice-subtitle { color:#c2eadb; font-size:14px; font-weight:650; }
.setup-choice-description { max-width:430px; color:#b1c7be; font-size:12px; line-height:1.55; }
.setup-choice-tags { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 6px; }
.setup-choice-tags span {
  padding:4px 7px; border:1px solid #3c6a5a; border-radius:999px;
  background:#182d28; color:#a7e8d0; font:600 9px/1.2 ui-monospace,monospace; letter-spacing:.04em;
}
[class*="st-key-setup_chosen_model_"] .architecture-diagram { width:100%; height:auto; filter:drop-shadow(0 8px 18px #0003); }
[class*="st-key-setup_summary_row_"] {
  margin:4px 0; padding:9px 12px; border:1px solid #344943; border-radius:11px;
  background:#17211f; transition:border-color .18s ease,background .18s ease;
}
[class*="st-key-setup_summary_row_"]:hover { border-color:#5a9985; background:#1b2b27; }
[class*="st-key-setup_summary_row_"] p { margin:0; font-size:12px; line-height:1.35; overflow-wrap:anywhere; }
[class*="st-key-setup_summary_row_"] strong { color:#dbeee6; font-weight:650; }
[class*="st-key-setup_summary_row_"] button {
  width:100%; min-height:34px; padding:0 8px; border:1px solid #487b6b; border-radius:9px;
  background:#213a32; color:#b1f1d9;
}
[class*="st-key-setup_summary_row_"] button:hover { border-color:#7bdbb9; background:#2a5042; }
[class*="st-key-setup_summary_row_"] button p { display:block; white-space:nowrap; font-size:11px; }
[role="dialog"] [data-testid="stMarkdownContainer"] h3 { margin:.15rem 0 .55rem; color:#f0f4f2; }
[role="dialog"] .architecture-diagram { display:block; width:100%; max-height:245px; margin:3px auto 8px; }
[role="dialog"] [data-testid="stTabs"] button { color:#b6c9c0; }
[role="dialog"] [data-testid="stTabs"] button[aria-selected="true"] { color:#b9f3d8; }
[role="dialog"] [role="radio"][aria-checked="true"] {
  border-color:#69cba5!important; background:#244638!important; color:#dcfff0!important;
  box-shadow:inset 0 0 0 1px #69cba522,0 0 14px #5ad1a51c;
}
.setup-editor-lead { margin:0 0 12px; padding:11px 13px; border:1px solid #354d47; border-radius:11px; background:#192320; color:#b7cbc3; font-size:12px; line-height:1.55; }
.setup-edit-hero {
  position:relative; display:grid; grid-template-columns:minmax(0,1.05fr) minmax(260px,.95fr); gap:18px;
  align-items:center; overflow:hidden; margin:3px 0 17px; padding:18px;
  border:1px solid #3d5b53; border-radius:17px;
  background:radial-gradient(ellipse at 8% -20%,#29463c 0%,transparent 48%),linear-gradient(130deg,#182522 0%,#1d2427 62%,#22232b 100%);
  box-shadow:inset 0 1px #ffffff0c,0 12px 30px #0003;
  animation:setup-edit-enter .42s cubic-bezier(.2,.8,.2,1) both;
}
.setup-edit-hero::after { content:""; position:absolute; width:210px; height:210px; right:-116px; top:-140px; border-radius:50%; background:#78d9b51b; filter:blur(4px); animation:setup-edit-halo 5s ease-in-out infinite alternate; pointer-events:none; }
.setup-edit-kicker { display:flex; align-items:center; gap:7px; color:#9be4c8; font:700 9px/1.4 ui-monospace,monospace; letter-spacing:.13em; }
.setup-edit-spark { display:grid; place-items:center; width:20px; height:20px; border:1px solid #477e68; border-radius:7px; color:#c8ffe3; background:#214537; font-size:12px; animation:setup-edit-spark 2.8s ease-in-out infinite; }
.setup-edit-title { margin:7px 0 3px; color:#f1f8f4; font-size:22px; font-weight:680; letter-spacing:-.035em; }
.setup-edit-description { max-width:390px; color:#a9bcb4; font-size:12px; line-height:1.5; }
.setup-edit-badges { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; margin-top:13px; }
.setup-edit-badges span { min-width:0; padding:7px 9px; border:1px solid #344c45; border-radius:9px; background:#151f1eaa; }
.setup-edit-badges small { display:block; color:#86b6a4; font:600 8px/1.3 ui-monospace,monospace; letter-spacing:.08em; }
.setup-edit-badges b { display:block; overflow:hidden; margin-top:2px; color:#deeee7; font:650 11px/1.35 ui-monospace,monospace; text-overflow:ellipsis; white-space:nowrap; }
.setup-edit-visual { position:relative; z-index:1; min-width:0; }
.setup-edit-visual svg { display:block; width:100%; height:auto; overflow:visible; filter:drop-shadow(0 8px 18px #0003); }
.setup-edit-plot-bg { fill:#111a1d; stroke:#344b46; stroke-width:1.2; }
.setup-edit-grid { fill:none; stroke:#283c39; stroke-width:1; stroke-dasharray:2 5; }
.setup-art-label { fill:#e2f3eb; font:700 10px ui-monospace,monospace; letter-spacing:.025em; }
.setup-art-muted { fill:#8fa99d; font:600 8px ui-monospace,monospace; letter-spacing:.055em; }
.setup-art-equation { fill:#c8d8ff; font:700 11px ui-monospace,monospace; }
.setup-art-axis { fill:none; stroke:#30453f; stroke-width:1.2; stroke-dasharray:3 5; }
.setup-art-wire { fill:none; stroke:#5ebda0; stroke-width:1.8; opacity:.62; }
.setup-art-flow { fill:none; stroke:#83e0c2; stroke-width:2; stroke-linecap:round; stroke-dasharray:6 5; animation:setup-art-flow 2.3s linear infinite; }
.setup-art-loop { fill:none; stroke:#9fe8ce; stroke-width:1.7; stroke-linecap:round; stroke-dasharray:4 5; animation:setup-art-flow 2s linear infinite; }
.setup-art-card { fill:#172321; stroke:#3a5a4f; stroke-width:1.2; transition:fill .2s ease,stroke .2s ease,filter .2s ease; }
.setup-art-card--active { fill:#1d3930; stroke:#71d1ad; filter:drop-shadow(0 0 8px #5ed0a42b); }
.setup-art-nodes circle { fill:#213e37; stroke:#78d3b4; stroke-width:1.5; }
.setup-art-pulse { fill:#b8ffe5; stroke:#6ed8b2; stroke-width:1.2; filter:url(#setup-edit-glow); animation:setup-edit-marker 2.2s ease-in-out infinite; }
.setup-art-pulse--late { animation-delay:1.1s; }
.setup-art-target { fill:none; stroke:#8594df; stroke-width:1.5; transition:stroke .2s ease,filter .2s ease; }
.setup-art-target--active { stroke:#d4c9ff; filter:drop-shadow(0 0 5px #aaa2ff88); }
.setup-art-target-dot { fill:#8e85da; }
.setup-art-search-dot { fill:#8898d9; stroke:#d1caff; stroke-width:.8; }
.setup-art-cycle { fill:#172321; stroke:#384e47; stroke-width:1.2; }
.setup-art-cycle--active { fill:#24483a; stroke:#75d2ad; filter:drop-shadow(0 0 5px #70d2aa39); }
.setup-art-column { fill:#172321; stroke:#59736d; stroke-width:1.2; }
.setup-art-time { stroke:#9a91df; }
.setup-art-state { stroke:#65cbaa; }
.setup-art-input { stroke:#67b9dc; }
.setup-art-clock-face { fill:#1a1c2c; stroke:#aaa0ef; stroke-width:1.8; }
.setup-art-clock-marks,.setup-art-clock-hands { fill:none; stroke:#c3baff; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.setup-art-clock-center { fill:#d8d1ff; }
.setup-art-signal-state,.setup-art-signal-input { fill:none; stroke-width:2; stroke-linecap:round; }
.setup-art-signal-state { stroke:#75dfb9; }
.setup-art-signal-input { stroke:#78c8e8; }
.setup-art-loss-area { fill:url(#setup-edit-fill); opacity:.45; }
.setup-art-loss-line { fill:none; stroke:#82e1bd; stroke-width:2.5; stroke-linecap:round; stroke-dasharray:460; stroke-dashoffset:460; animation:setup-edit-draw 1.6s .12s ease-out forwards; }
.setup-art-network-wire { stroke:#56a98c; stroke-width:1; opacity:.23; }
.setup-art-network-node { fill:#203d35; stroke:#78d9b4; stroke-width:1.3; animation:setup-art-node-pulse 2.5s ease-in-out infinite; }
.setup-art-network-node--0,.setup-art-network-node--3 { fill:#28304a; stroke:#aca1ef; }
.setup-art-network-node--2 { animation-delay:.35s; }
.setup-art-plant { fill:none; stroke:#a6dfca; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
.setup-art-gauge-track { fill:none; stroke:#324941; stroke-width:8; stroke-linecap:round; }
.setup-art-gauge-fit { fill:none; stroke:#75d8b2; stroke-width:8; stroke-linecap:round; transition:stroke-dasharray .25s ease; }
.setup-art-gauge-speed { fill:none; stroke:#a69be9; stroke-width:8; stroke-linecap:round; transition:stroke-dasharray .25s ease; }
.setup-art-bracket { fill:none; stroke:#a49be7; stroke-width:1.5; stroke-linecap:round; }
.setup-edit-visual--physics .setup-edit-plot-bg { stroke:#45636a; }
.setup-edit-visual--context .setup-edit-plot-bg { stroke:#65584b; }
.setup-edit-visual--goal .setup-edit-plot-bg,.setup-edit-visual--effort .setup-edit-plot-bg { stroke:#514d75; }
.setup-edit-visual--signals .setup-edit-plot-bg { stroke:#3e6372; }
.setup-edit-visual--epochs .setup-edit-plot-bg,.setup-edit-visual--training .setup-edit-plot-bg { stroke:#496651; }
.setup-edit-visual--targets .setup-edit-plot-bg { stroke:#565184; }
.setup-edit-visual--advanced .setup-edit-plot-bg { stroke:#585478; }
@keyframes setup-art-flow { to { stroke-dashoffset:-33; } }
@keyframes setup-art-node-pulse { 0%,100% { filter:drop-shadow(0 0 1px #72dbb44a); } 50% { filter:drop-shadow(0 0 6px #72dbb4a3); } }
.setup-edit-area { fill:url(#setup-edit-fill); opacity:0; animation:setup-edit-area .7s .25s ease-out forwards; }
.setup-edit-signal,.setup-edit-forecast { fill:none; stroke:#72dec0; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:500; stroke-dashoffset:500; animation:setup-edit-draw 1.5s .12s cubic-bezier(.3,.7,.2,1) forwards; }
.setup-edit-forecast { stroke:#b8adff; stroke-width:2; stroke-dasharray:5 5; stroke-dashoffset:0; animation:setup-edit-dash 2s linear infinite; }
.setup-edit-marker { fill:#d8fff0; filter:url(#setup-edit-glow); animation:setup-edit-marker 2.4s 1s ease-in-out infinite; }
.setup-edit-orbit { fill:none; stroke:#84e2c6; stroke-width:1; opacity:.6; transform-box:fill-box; transform-origin:center; animation:setup-edit-orbit 2.4s 1s ease-in-out infinite; }
.setup-edit-axis { fill:#89a99d; font:600 8px ui-monospace,monospace; letter-spacing:.08em; }
.setup-edit-visual-caption { display:flex; align-items:center; justify-content:flex-end; gap:7px; margin-top:3px; color:#8ea79d; font:10px/1.4 ui-monospace,monospace; }
.setup-edit-agent-logic { display:grid; gap:3px; margin-top:9px; padding:8px 10px; border-left:2px solid #68c6a1; border-radius:0 8px 8px 0; background:#13201dcc; color:#afc6ba; font-size:10px; line-height:1.45; }
.setup-edit-agent-logic small { color:#87d6b4; font:700 8px/1.3 ui-monospace,monospace; letter-spacing:.1em; }
.setup-edit-live-dot { width:6px; height:6px; border-radius:50%; background:#74dfbc; box-shadow:0 0 9px #74dfbc88; animation:setup-edit-live 1.7s ease-in-out infinite; }
.setup-edit-hero--effort,.setup-edit-hero--cycles { border-color:#554f79; background:radial-gradient(ellipse at 88% 0%,#3b315826 0%,transparent 48%),radial-gradient(ellipse at 8% -20%,#29463c 0%,transparent 48%),linear-gradient(130deg,#182522 0%,#1d2427 62%,#22232b 100%); }
.setup-edit-hero--physics { border-color:#416675; background:radial-gradient(ellipse at 88% 0%,#27607026 0%,transparent 48%),linear-gradient(130deg,#182522 0%,#1d2427 62%,#22232b 100%); }
@keyframes setup-edit-enter { from { opacity:0; transform:translateY(7px) scale(.992); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes setup-edit-halo { to { transform:translate(-26px,22px) scale(1.18); opacity:.55; } }
@keyframes setup-edit-spark { 0%,100% { box-shadow:0 0 0 #80e4bd00; transform:rotate(0); } 50% { box-shadow:0 0 16px #80e4bd4a; transform:rotate(12deg); } }
@keyframes setup-edit-area { to { opacity:1; } }
@keyframes setup-edit-draw { to { stroke-dashoffset:0; } }
@keyframes setup-edit-dash { from { stroke-dashoffset:0; } to { stroke-dashoffset:-30; } }
@keyframes setup-edit-marker { 0%,100% { opacity:.78; r:4; } 50% { opacity:1; r:5.5; } }
@keyframes setup-edit-orbit { 0%,100% { transform:scale(.8); opacity:.24; } 50% { transform:scale(1.2); opacity:.75; } }
@keyframes setup-edit-live { 0%,100% { opacity:.55; transform:scale(.8); } 50% { opacity:1; transform:scale(1.22); } }
.setup-editor-current,.setup-editor-note {
  margin:10px 0; padding:11px 13px; border:1px solid #365c50; border-radius:11px;
  background:linear-gradient(120deg,#1c332c,#1d2427); color:#aec9bd; font-size:12px; line-height:1.5;
}
.setup-editor-current small { display:block; margin-bottom:4px; color:#81d9b9; font:700 9px/1.3 ui-monospace,monospace; letter-spacing:.1em; }
.setup-editor-current strong { display:block; color:#effbf5; font-size:15px; }
.setup-editor-current span { display:block; margin-top:2px; color:#9db7ab; }
.setup-editor-note b { color:#e6f6ed; }
.setup-network-preview { margin:12px 0; padding:12px 10px 7px; border:1px solid #365a50; border-radius:13px; background:linear-gradient(145deg,#121e1d,#182522); overflow:hidden; }
.setup-network-title { margin:0 0 4px 5px; color:#83d8b8; font:700 9px/1.4 ui-monospace,monospace; letter-spacing:.13em; }
.setup-network-preview svg { display:block; width:100%; height:auto; }
.setup-edit-visual--network .setup-edit-nn-kicker { fill:#86d9b9; font:700 7px ui-monospace,monospace; letter-spacing:.12em; }
.setup-edit-visual--network .setup-edit-nn-label { fill:#9ab7ac; font:700 7px ui-monospace,monospace; letter-spacing:.02em; }
.setup-edit-visual--network .setup-edit-nn-width { fill:#e1f6ec; font:700 9px ui-monospace,monospace; }
.setup-edit-visual--network .setup-net-wire { stroke-dasharray:4 6; animation:setup-edit-dash 3s linear infinite; }
.setup-net-edge { stroke:#55a98d; stroke-opacity:.4; stroke-width:1; }
.setup-net-node { fill:#263c36; stroke:#70d5ae; stroke-width:1.6; }
.setup-net-node.input { fill:#26343c; stroke:#81b5dd; }
.setup-net-node.output { fill:#342f48; stroke:#a79be7; }
.setup-network-preview text { fill:#a9c4b9; font:700 9px ui-monospace,monospace; letter-spacing:.03em; }
.setup-network-preview text.setup-net-width { fill:#e3f6ed; font-size:10px; }
.setup-history-visual { display:flex; align-items:center; justify-content:center; gap:6px; flex-wrap:wrap; margin:13px 0; padding:13px; border:1px solid #31594c; border-radius:13px; background:linear-gradient(120deg,#16231f,#1b2428); }
.setup-history-step { display:flex; flex-direction:column; align-items:center; gap:4px; min-width:43px; padding:8px 7px; border:1px solid #35584d; border-radius:9px; background:#1c3029; }
.setup-history-step small { color:#9eb8ae; font:9px ui-monospace,monospace; }
.setup-history-step b { color:#ceeee0; font:700 10px ui-monospace,monospace; }
.setup-history-step.current { border-color:#71cda9; background:#244b3b; box-shadow:0 0 14px #57d0a333; }
.setup-history-ellipsis,.setup-history-arrow { color:#79d3b1; font:700 18px ui-monospace,monospace; }
.setup-history-output { padding:9px 12px; border:1px solid #6963a5; border-radius:9px; background:#292941; color:#d0caff; font:700 11px ui-monospace,monospace; }
.setup-effort-options { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin:13px 0; }
.setup-effort-card { display:flex; flex-direction:column; gap:4px; padding:12px; border:1px solid #364943; border-radius:11px; background:#1a2422; }
.setup-effort-card small { color:#91aa9f; font:700 9px ui-monospace,monospace; letter-spacing:.08em; }
.setup-effort-card strong { color:#d4e9df; font-size:14px; }
.setup-effort-card span { color:#91a79d; font-size:10px; }
.setup-effort-card.selected { border-color:#75cbaa; background:linear-gradient(135deg,#234537,#1d2c2a); box-shadow:inset 0 0 0 1px #69c39c22,0 0 14px #51cb9b18; }
.setup-effort-card.selected small,.setup-effort-card.selected strong { color:#c3f5dd; }
[role="dialog"] [data-testid="stVerticalBlockBorderWrapper"]:has(input) { border-color:#394943; border-radius:12px; background:#1b2322; }
@media(max-width:650px) {
  .setup-edit-hero { grid-template-columns:1fr; gap:11px; padding:14px; }
  .setup-edit-title { font-size:19px; }
  .setup-edit-visual { max-width:520px; }
  .setup-edit-visual-caption { justify-content:flex-start; }
  [class*="st-key-setup_chosen_model_"] [data-testid="stHorizontalBlock"] { flex-direction:column; }
  .setup-effort-options { grid-template-columns:1fr; }
}
@media(prefers-reduced-motion:reduce) {
  .setup-edit-hero,.setup-edit-hero::after,.setup-edit-spark,.setup-edit-area,.setup-edit-signal,
  .setup-edit-forecast,.setup-edit-marker,.setup-edit-orbit,.setup-edit-live,.setup-art-flow,
  .setup-art-loop,.setup-art-pulse,.setup-art-loss-line,.setup-art-network-node,
  .setup-edit-visual--network .setup-net-wire { animation:none!important; }
  .setup-edit-area { opacity:1; }
  .setup-edit-signal { stroke-dashoffset:0; }
  .setup-art-loss-line { stroke-dashoffset:0; }
}
.setup-network-shell {
  margin:14px 0 10px; padding:12px; border:1px solid #3b6258; border-radius:18px;
  background:linear-gradient(145deg,#15211f,#182124 58%,#1c2028);
  box-shadow:inset 0 1px #ffffff0e,0 10px 28px #0003;
}
.setup-network-heading,.setup-network-foot {
  display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;
}
.setup-network-heading { padding:3px 6px 11px; }
.setup-network-kicker { color:#84d8be; font:700 10px/1.3 ui-monospace,monospace; letter-spacing:.13em; }
.setup-network-meta { color:#d9ede4; font:600 11px/1.3 ui-monospace,monospace; }
.setup-network-svg { display:block; width:100%; height:auto; overflow:visible; }
.setup-net-ground { fill:#111b1d; stroke:#2d4c47; stroke-width:1.2; }
.setup-net-axis { fill:none; stroke:#31594e; stroke-width:1.2; stroke-dasharray:3 9; }
.setup-net-wire { fill:none; stroke:#69bfa5; stroke-width:1.1; opacity:.32; }
.setup-net-layer { fill:#1c3430; stroke:#4f9e89; stroke-width:1.6; filter:drop-shadow(0 2px 10px #47b7952b); }
.setup-net-node { stroke-width:1.6; }
.setup-net-node.input { fill:#265260; stroke:#7bc5d7; }
.setup-net-node.hidden { fill:#60ccab; stroke:#bbffe5; filter:drop-shadow(0 0 5px #6adab18c); animation:setup-node-breathe 2.5s ease-in-out infinite; }
.setup-net-node.output { fill:#736bb8; stroke:#c8bfff; filter:drop-shadow(0 0 5px #a99bff77); }
.setup-net-loop { fill:none; stroke:#9be8cb; stroke-width:2; stroke-linecap:round; stroke-dasharray:5 5; animation:setup-loop-flow 2s linear infinite; }
.setup-net-label { fill:#e1f2ec; font:700 12px ui-monospace,monospace; }
.setup-net-count { fill:#9fbbb1; font:11px ui-monospace,monospace; }
.setup-network-foot { padding:10px 6px 2px; color:#9dbbb0; font:11px/1.45 ui-monospace,monospace; }
.setup-network-metrics {
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; margin-top:10px;
}
.setup-network-metrics span {
  min-width:0; padding:8px 10px; border:1px solid #315249; border-radius:9px;
  background:#152925; box-shadow:inset 0 1px #ffffff08;
}
.setup-network-metrics small {
  display:block; color:#8fb9aa; font:600 9px/1.25 ui-monospace,monospace; letter-spacing:.06em;
}
.setup-network-metrics strong {
  display:block; margin-top:3px; overflow:hidden; text-overflow:ellipsis;
  color:#e4f5ee; font:700 13px/1.2 ui-monospace,monospace;
}
@keyframes setup-node-breathe { 0%,100% { opacity:.8; } 50% { opacity:1; } }
@keyframes setup-loop-flow { to { stroke-dashoffset:-20; } }
@media(max-width:620px) {
  .setup-network-shell { padding:7px; }
  .setup-network-kicker,.setup-network-meta { font-size:9px; }
  .setup-network-foot { font-size:10px; }
  .setup-network-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media(prefers-reduced-motion:reduce) {
  .setup-net-node.hidden,.setup-net-loop { animation:none!important; }
}
@keyframes arch-neuron-pulse { 0%,100% { filter:drop-shadow(0 0 1px #70e3c455); } 50% { filter:drop-shadow(0 0 5px #70e3c4bb); } }
@keyframes arch-signal-flow { to { stroke-dashoffset:-18; } }
[class*="st-key-setup_pinn_panel_"] [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3b4c4b; border-radius:14px;
  background:linear-gradient(120deg,#1d2929 0%,#202326 65%,#24242a 100%);
}
[class*="st-key-setup_pinn_panel_"] [data-testid="stCaptionContainer"] p {
  color:#9badaa; font-size:12px; line-height:1.55;
}
.st-key-conversation_file_panel,
.st-key-conversation_file_panel_expanded {
  position:fixed!important; inset:0 0 0 auto!important;
  z-index:1005; width:290px!important; min-width:250px!important; max-width:100vw!important;
  height:100dvh!important; max-height:none!important; min-height:0!important;
  display:flex!important; flex-direction:column!important;
  box-sizing:border-box; padding:14px 12px 18px;
  border:0; border-left:1px solid #46524f; border-radius:0;
  background:linear-gradient(180deg,#202826 0%,#1d2224 58%,#202027 100%);
  box-shadow:-14px 0 34px #0005,inset 1px 0 #ffffff08;
  overflow-x:hidden!important; overflow-y:auto!important;
  scrollbar-color:#71847b transparent; scrollbar-width:thin; overscroll-behavior:contain;
}
.st-key-conversation_file_panel [data-testid="stVerticalBlock"],
.st-key-conversation_file_panel_expanded [data-testid="stVerticalBlock"] { gap:.35rem!important; }
.st-key-conversation_file_panel h4,
.st-key-conversation_file_panel_expanded h4 { margin:.15rem 0!important; font-size:15px!important; }
.st-key-conversation_file_panel [data-testid="stCaptionContainer"] p,
.st-key-conversation_file_panel_expanded [data-testid="stCaptionContainer"] p {
  margin:0!important; font-size:10px!important; line-height:1.35!important;
}
.st-key-conversation_file_panel button,
.st-key-conversation_file_panel_expanded button { min-height:32px!important; padding:5px 8px!important; }
.st-key-conversation_file_panel_expanded {
  inset:12px!important; width:auto!important; height:calc(100dvh - 24px)!important;
  border:1px solid #52615c; border-radius:16px;
  box-shadow:0 18px 48px #0008,inset 0 1px #ffffff0b;
}
.st-key-conversation_file_panel > [data-testid="stLayoutWrapper"]:first-child,
.st-key-conversation_file_panel_expanded > [data-testid="stLayoutWrapper"]:first-child {
  position:sticky!important; top:-14px!important; z-index:5;
  margin:-14px -12px 0!important; padding:10px 12px 9px!important;
  background:linear-gradient(180deg,#202826 0%,#1d2224 100%)!important;
  border-bottom:1px solid #384542;
}
.st-key-conversation_file_panel_expanded > [data-testid="stLayoutWrapper"]:first-child {
  margin:-14px -12px 0!important; border-radius:15px 15px 5px 5px;
}
.st-key-conversation_file_panel::-webkit-scrollbar,
.st-key-conversation_file_panel_expanded::-webkit-scrollbar { width:8px; }
.st-key-conversation_file_panel::-webkit-scrollbar-thumb,
.st-key-conversation_file_panel_expanded::-webkit-scrollbar-thumb {
  border:2px solid transparent; border-radius:8px; background:#71847b content-box;
}
[class*="st-key-run_files_"] [data-testid="stVerticalBlock"] { gap:.45rem; }
[class*="st-key-chat_file_card"] [data-testid="stVerticalBlockBorderWrapper"] {
  padding:4px 8px; border-color:#393939; background:#252525; border-radius:10px;
}
[class*="st-key-chat_file_card"] { width:100%; max-width:440px; margin-right:auto; }
[class*="st-key-chat_file_card"] [data-testid="stVerticalBlock"] { gap:.05rem; }
[class*="st-key-chat_file_card"] button {
  height:28px; min-height:28px; padding:2px 4px; border:0; background:transparent;
  text-align:left; justify-content:flex-start;
}
[class*="st-key-chat_file_card"] button:hover { background:#303030; }
[class*="st-key-chat_file_card"] [data-testid="stButton"] p {
  overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
}
[class*="st-key-chat_file_card"] [data-testid="stCaptionContainer"] p {
  margin:0; font-size:10px; line-height:1.25;
}
[class*="st-key-chat_file_icon_"] button {
  position:relative; display:grid; place-items:center; width:36px; height:36px; min-height:36px; padding:0;
  border:0; border-radius:9px; color:#c3c9ff; background:#343445;
}
[class*="st-key-chat_file_icon_"] button [data-testid="stIconMaterial"] {
  position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
}
[class*="st-key-chat_file_icon_"] button p { font-size:0; }
[class*="st-key-chat_file_icon_"] button:hover { background:#41415a; }
[class*="st-key-open_chat_file_action_"] button {
  height:30px; min-height:30px; padding:4px 12px; border:1px solid #424242;
  border-radius:999px; background:transparent; text-align:center; justify-content:center;
}
[class*="st-key-open_chat_file_action_"] button:hover { border-color:#727272; background:#303030; }
.st-key-conversation_live_run_card [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3d514d; border-radius:18px;
  background:radial-gradient(ellipse at 8% 0%,#263b37 0%,#242626 42%,#222222 100%);
  box-shadow:0 12px 34px #0003,inset 0 1px #ffffff08;
}
.run-learning-snapshot {
  overflow:hidden; margin:8px 0 2px; padding:14px 16px 16px; border:1px solid #3c5750; border-radius:17px;
  background:radial-gradient(ellipse at 10% 0%,#213630 0%,transparent 48%),linear-gradient(145deg,#171e1d 0%,#1b2022 69%,#22212b 100%);
  box-shadow:inset 0 1px #ffffff08,0 10px 26px #0002; animation:run-tour-enter .35s ease both;
}
.run-tour-heading { display:flex; align-items:center; justify-content:space-between; gap:8px; color:#83cdb3; font:700 8px/1.4 ui-monospace,monospace; letter-spacing:.12em; }
.run-tour-layout { display:grid; grid-template-columns:minmax(0,1.2fr) minmax(230px,.8fr); align-items:stretch; gap:14px; margin-top:11px; }
.run-tour-art { display:flex; flex-direction:column; justify-content:center; min-width:0; min-height:270px; padding:11px 13px 8px; border:1px solid #344943; border-radius:13px; background:radial-gradient(ellipse at 50% 100%,#1b302d,#141a1c 72%); }
.run-tour-title { margin:0 0 3px 4px; color:#e5f2ec; font-size:13px; font-weight:680; letter-spacing:-.01em; }
.run-tour-illustration { display:grid; place-items:center; width:100%; min-width:0; }
.run-tour-svg { display:block; width:100%; height:auto; max-height:255px; overflow:visible; }
.run-tour-story { display:flex; flex-direction:column; justify-content:center; gap:10px; min-width:0; }
.run-tour-general,.run-tour-specific { padding:12px 13px; border:1px solid #33413d; border-radius:12px; background:#171e1d; }
.run-tour-specific { border-color:#405d53; background:linear-gradient(145deg,#1b2a25,#1c2224); }
.run-tour-general small,.run-tour-specific small { display:block; color:#8ebaa9; font:700 8px/1.4 ui-monospace,monospace; letter-spacing:.12em; }
.run-tour-specific small { color:#8ddbbd; }
.run-tour-general p,.run-tour-specific p { margin:6px 0 0; color:#b8c9c1; font-size:12px; line-height:1.6; overflow-wrap:anywhere; }
.run-tour-specific p { color:#e0ebe5; }
.run-tour-story--current { justify-content:center; }
.run-tour-current-only { padding:15px; }
.run-tour-current-only p { font-size:13px; }
.run-tour-current-meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
.run-tour-current-meta span { min-width:0; padding:8px 9px; border:1px solid #334641; border-radius:9px; background:#171e1d; color:#b9d0c7; font:600 10px/1.4 ui-monospace,monospace; overflow-wrap:anywhere; }
.run-tour-current-meta span:first-child { color:#9de5c5; border-color:#456c5b; background:#1b2b25; }
.run-tour-runplot-line { fill:none; stroke:#77dfbc; stroke-width:2.8; stroke-linecap:round; stroke-linejoin:round; filter:drop-shadow(0 0 5px #65d6b34a); }
.run-tour-runplot-dot { fill:#baffec; stroke:#65cda9; stroke-width:1.5; }
.run-tour-runplot-dot--best { fill:#b7aaff; stroke:#9388e8; stroke-width:2; filter:drop-shadow(0 0 5px #a59af566); }
.st-key-run_tour_dots { margin:0 0 3px; }
.st-key-run_tour_dots [data-testid="stHorizontalBlock"] { justify-content:center!important; gap:5px!important; }
.st-key-run_tour_dots [data-testid="stElementContainer"] { flex:0 0 23px!important; width:23px!important; }
.st-key-run_tour_dots button { width:24px!important; min-width:24px!important; height:24px!important; min-height:24px!important; padding:0!important; border:1px solid transparent!important; border-radius:50%!important; background:transparent!important; font-size:15px!important; line-height:1!important; box-shadow:none!important; }
.st-key-run_tour_dots [data-testid="stBaseButton-primary"] { color:#8ce8c5!important; text-shadow:0 0 9px #70d9b5bb; }
.st-key-run_tour_dots [data-testid="stBaseButton-tertiary"] { color:#657a71!important; }
.st-key-run_tour_dots button:hover { color:#d8fff0!important; transform:scale(1.2); }
.run-tour-grid { fill:none; stroke:#2a3c39; stroke-width:1; stroke-dasharray:2 5; }
.run-tour-axis { fill:none; stroke:#53665f; stroke-width:1.2; stroke-linecap:round; }
.run-tour-signal { fill:none; stroke:#6ed2b4; stroke-width:2.7; stroke-linecap:round; stroke-linejoin:round; filter:drop-shadow(0 0 4px #65d6b33d); }
.run-tour-prediction { fill:none; stroke:#aaa1f2; stroke-width:2.3; stroke-linecap:round; stroke-dasharray:5 5; animation:run-tour-dash 2.1s linear infinite; }
.run-tour-input-wave { fill:none; stroke:#73bce7; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:5 5; animation:run-tour-dash 2.5s linear infinite; }
.run-tour-signal--purple { fill:none; stroke:#b49bea; stroke-width:2.4; stroke-linecap:round; }
.run-tour-boundary { fill:none; stroke:#9b92d9; stroke-width:1.4; stroke-dasharray:4 4; }
.run-tour-sample-ticks { fill:none; stroke:#526961; stroke-width:1.2; }
.run-tour-sample { fill:#d2fff0; stroke:#6bd7b7; stroke-width:1.5; filter:drop-shadow(0 0 5px #6ad7b9b3); }
.run-tour-wire { fill:none; stroke:#4b7a6c; stroke-width:1.2; stroke-dasharray:4 4; animation:run-tour-dash 2.8s linear infinite; }
.run-tour-block { fill:#1b2c2b; stroke:#4c8c7e; stroke-width:1.2; }
.run-tour-block--model { fill:#18372f; stroke:#58ad91; }
.run-tour-block--output { fill:#25243c; stroke:#8580c5; }
.run-tour-equation { fill:#25243a; stroke:#928add; stroke-width:1.3; }
.run-tour-equation--off { opacity:.42; stroke-dasharray:4 4; }
.run-tour-equation-text { fill:#d1cbff; font:700 15px ui-monospace,monospace; }
.run-tour-neuron { fill:#315b50; stroke:#83dec0; stroke-width:1.3; }
.run-tour-neuron--active { fill:#78e0bd; filter:drop-shadow(0 0 6px #77e2c199); animation:run-tour-glow 1.8s ease-in-out infinite; }
.run-tour-step rect { fill:#182625; stroke:#45675e; stroke-width:1.2; }
.run-tour-step text { fill:#d5ebe2; font:700 9px ui-monospace,monospace; }
.run-tour-step .run-tour-tiny { fill:#8eaaa0; font-size:7px; }
.run-tour-step--current rect { fill:#204438; stroke:#67c69f; filter:drop-shadow(0 0 6px #68d1a936); }
.run-tour-memory { fill:none; stroke:#a3f0da; stroke-width:2.5; stroke-linecap:round; stroke-dasharray:5 4; animation:run-tour-dash 2.2s linear infinite; }
.run-tour-heldout { fill:#8883ce; opacity:.09; stroke:#aaa1f2; stroke-width:1; stroke-dasharray:3 4; }
.run-tour-label { fill:#a6bdb3; font:700 8px ui-monospace,monospace; letter-spacing:.1em; }
.run-tour-note { fill:#c5bdf5; font:700 7px ui-monospace,monospace; letter-spacing:.07em; }
.run-tour-tiny { fill:#91a89e; font:600 8px ui-monospace,monospace; letter-spacing:.04em; }
.run-tour-legend { fill:#a5b4ac; font:7px Arial,sans-serif; }
.run-tour-legend-observed { stroke:#6ed2b4; stroke-width:2; }
.run-tour-legend-predicted { stroke:#aaa1f2; stroke-width:2; stroke-dasharray:3 2; }
@keyframes run-tour-enter { from { opacity:.65; transform:translateY(3px); } to { opacity:1; transform:translateY(0); } }
@keyframes run-tour-dash { to { stroke-dashoffset:-24; } }
@keyframes run-tour-glow { 0%,100% { filter:drop-shadow(0 0 2px #77e2c166); } 50% { filter:drop-shadow(0 0 8px #77e2c1bb); } }
.st-key-conversation_live_run_card [data-testid="stExpander"] { margin-top:8px; }
.st-key-conversation_live_run_card [data-testid="stMarkdownContainer"] h3 {
  margin:.4rem 0 .2rem; font-size:1.2rem; letter-spacing:-.025em; color:#eef4f1;
}
.run-live-label { display:flex; align-items:center; gap:9px; color:#cce9df; font-size:13px; font-weight:650; }
.run-live-dot { width:9px; height:9px; border-radius:50%; background:#68d6ad; box-shadow:0 0 0 4px #68d6ad20; animation:run-live-pulse 1.8s ease-in-out infinite; }
@keyframes run-live-pulse { 0%,100% { box-shadow:0 0 0 3px #68d6ad18; opacity:1; } 50% { box-shadow:0 0 0 7px #68d6ad08; opacity:.72; } }
.run-process-diagram { margin:5px 0 13px; padding:11px; border:1px solid #3b504a; border-radius:14px; background:linear-gradient(135deg,#17211f,#1e2222 62%,#211f29); box-shadow:inset 0 1px #ffffff08; }
.st-key-run_details_card { margin-top:8px; }
.st-key-conversation_live_run_card [data-testid="stStatusWidget"] {
  display:block!important; visibility:visible!important; margin-top:4px;
  border:1px solid #3b4843; border-radius:12px;
  background:linear-gradient(135deg,#1a2220,#202225);
}
.st-key-conversation_live_run_card [data-testid="stStatusWidget"] summary { padding:9px 12px; color:#dce9e3; }
.st-key-conversation_live_run_card [data-testid="stStatusWidget"] summary:hover { background:#ffffff05; }
[class*="st-key-message_activity_"] [data-testid="stStatusWidget"] {
  display:block!important; visibility:visible!important; margin:5px 0 9px;
  border:1px solid #383b3a; border-radius:11px; background:#252626;
}
[class*="st-key-message_activity_"] [data-testid="stStatusWidget"] summary,
[class*="st-key-completed_thinking_"] [data-testid="stStatusWidget"] summary {
  padding:7px 11px; color:#b9c9c2; font-size:14px; line-height:1.45;
}
[class*="st-key-message_activity_"] [data-testid="stStatusWidget"] summary:hover { background:#ffffff05; }
[class*="st-key-message_activity_"] [data-testid="stCaptionContainer"] p,
[class*="st-key-completed_thinking_"] [data-testid="stCaptionContainer"] p {
  color:#a5aaa7; font-size:13px; line-height:1.5;
}
[class*="st-key-message_activity_"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-message_activity_"] li,
[class*="st-key-completed_thinking_"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-completed_thinking_"] li {
  font-size:14px!important; line-height:1.55!important;
}
[class*="st-key-message_activity_"] ul,
[class*="st-key-completed_thinking_"] ul { margin:.2rem 0 .4rem!important; padding-left:1.15rem!important; }
[class*="st-key-message_activity_"] hr { margin:.2rem 0 .7rem!important; border-color:#454a47!important; opacity:.8; }
[class*="st-key-message_activity_"] li,
[class*="st-key-completed_thinking_"] li { margin:.14rem 0!important; }
[class*="st-key-completed_thinking_"] [data-testid="stStatusWidget"] {
  display:block!important; visibility:visible!important; margin:5px 0 9px;
  border:1px solid #3b4843; border-radius:12px;
  background:linear-gradient(135deg,#1a2220,#202225);
}
[class*="st-key-completed_thinking_"] [data-testid="stStatusWidget"] summary { padding:9px 12px; color:#dce9e3; }
[class*="st-key-completed_thinking_"] [data-testid="stStatusWidget"] summary:hover { background:#ffffff05; }
[class*="st-key-live_cycle_mse_"] [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#3d514a; border-radius:15px;
  background:linear-gradient(145deg,#192320 0%,#202323 62%,#21212a 100%);
  box-shadow:inset 0 1px #ffffff08,0 10px 24px #0002;
}
[class*="st-key-live_cycle_mse_"] [data-testid="stMarkdownContainer"] p:first-child {
  color:#e7f1ec; font-size:14px; font-weight:650;
}
[class*="st-key-live_cycle_mse_"] [data-testid="stMetric"] {
  min-height:68px; padding:9px 11px; border:1px solid #384641; border-radius:11px; background:#171d1caa;
}
[class*="st-key-live_cycle_mse_"] [data-testid="stMetricLabel"] p { color:#a5b3ad; font-size:10px; }
[class*="st-key-live_cycle_mse_"] [data-testid="stMetricValue"] { color:#d8f4e8; font-size:1rem; }
.run-mse-empty { overflow:hidden; border:1px solid #34443f; border-radius:12px; background:linear-gradient(180deg,#171d1c,#191b20); }
.run-mse-empty svg { display:block; width:100%; height:auto; }
.run-mse-empty svg path { fill:none; stroke:#2b3b37; stroke-width:1; stroke-dasharray:3 5; }
.run-mse-empty svg .run-mse-empty-axis { stroke:#62756e; stroke-dasharray:none; }
.run-mse-empty svg text { fill:#82968e; font:600 9px ui-monospace,monospace; letter-spacing:.08em; }
.run-mse-empty svg .run-mse-empty-note { fill:#9eb7ac; font:700 10px ui-monospace,monospace; }
.run-process-heading { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:0 2px 9px; color:#8fa99f; font:700 9px/1.35 ui-monospace,monospace; letter-spacing:.1em; }
.run-process-heading b { color:#c7d9d2; font:600 10px/1.35 ui-monospace,monospace; letter-spacing:0; }
.run-process-flow { display:grid; grid-template-columns:minmax(76px,.8fr) 14px minmax(86px,.9fr) 14px minmax(218px,2.2fr) 14px minmax(82px,.85fr) 14px minmax(78px,.8fr); align-items:center; gap:5px; }
.run-flow-block { display:flex; flex-direction:column; justify-content:center; min-width:0; min-height:77px; padding:9px 10px; border:1px solid #373e3b; border-radius:11px; background:#202423; transition:border-color .2s ease,background .2s ease,box-shadow .2s ease; }
.run-flow-block small,.run-search-heading small { color:#82928b; font:700 8px/1.35 ui-monospace,monospace; letter-spacing:.07em; }
.run-flow-block b { margin-top:4px; color:#d8e5df; font-size:12px; line-height:1.2; }
.run-flow-block span { margin-top:4px; overflow:hidden; color:#929e99; font-size:9px; line-height:1.3; text-overflow:ellipsis; white-space:nowrap; }
.run-flow-block--done { border-color:#36584b; background:linear-gradient(145deg,#1c332b,#202725); }
.run-flow-block--done small { color:#80c9a9; }
.run-flow-block--active { border-color:#70c8a6; background:linear-gradient(145deg,#26483a,#222c29); box-shadow:0 0 18px #58c99c18,inset 0 0 0 1px #78ddb21a; }
.run-flow-block--active small { color:#9ae2c2; }
.run-flow-block--active b { color:#f0fff7; }
.run-process-arrow { color:#72cba9; text-align:center; font-size:16px; line-height:1; }
.run-search-block { min-width:0; padding:9px 10px 8px; border:1px solid #45404f; border-radius:12px; background:linear-gradient(145deg,#24222e,#202321); }
.run-search-block--active { border-color:#60a990; background:linear-gradient(145deg,#203c33,#24252a); box-shadow:0 0 18px #53c39818; }
.run-search-block--done { border-color:#36584b; background:linear-gradient(145deg,#1c302a,#202422); }
.run-search-heading { display:flex; align-items:center; justify-content:space-between; gap:5px; margin-bottom:7px; }
.run-search-heading small { color:#b4a9e6; }
.run-search-heading b { color:#b8c9c1; font:600 9px/1.35 ui-monospace,monospace; white-space:nowrap; }
.run-search-agents { display:grid; grid-template-columns:minmax(0,1fr) 11px minmax(0,.84fr) 11px minmax(0,1fr); align-items:center; gap:3px; }
.run-search-agents > span { color:#77c8a8; text-align:center; font-size:12px; }
.run-search-agent,.run-search-score { display:flex; flex-direction:column; justify-content:center; min-width:0; min-height:38px; padding:5px 6px; border:1px solid #3b4943; border-radius:8px; background:#1b2522; }
.run-search-agent small,.run-search-score small { overflow:hidden; color:#8da69b; font:700 7px/1.3 ui-monospace,monospace; letter-spacing:.05em; text-overflow:ellipsis; white-space:nowrap; }
.run-search-agent b,.run-search-score b { margin-top:2px; color:#d9e9e1; font-size:10px; line-height:1.1; }
.run-search-score { border-color:#514a6a; background:#26243a; }
.run-search-score small { color:#b5a9ed; }
.run-search-agent--active { border-color:#71c7a4; background:#27473a; box-shadow:inset 0 0 0 1px #73d5ae1c; }
.run-search-agent--active small { color:#a4e5c6; }
.run-search-feedback { margin-top:6px; padding-top:5px; border-top:1px dashed #4b5d54; color:#adcfbf; font-size:8px; line-height:1.25; text-align:center; }
.run-critic-title { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:1px 0 10px; }
.run-critic-title b { color:#edf8f2; font-size:13px; }
.run-critic-title span { padding:3px 7px; border:1px solid #514b6b; border-radius:99px; background:#29263a; color:#cbc3f0; font:700 9px/1.3 ui-monospace,monospace; }
.run-critic-section-label,.run-critic-next small { display:block; margin:9px 0 4px; color:#86cdb0; font:700 8px/1.35 ui-monospace,monospace; letter-spacing:.09em; }
.run-critic-reasoning { padding:9px 10px; border-left:2px solid #6ec5a1; border-radius:0 8px 8px 0; background:#182420; color:#c2d4cb; font-size:12px; line-height:1.55; white-space:pre-wrap; overflow-wrap:anywhere; }
.run-critic-next { display:flex; flex-direction:column; gap:2px; margin-top:8px; padding:8px 10px; border:1px solid #3f5149; border-radius:9px; background:#202a26; }
.run-critic-next small { margin:0; }
.run-critic-next span { color:#dae9e1; font-size:11px; line-height:1.4; }
.st-key-conversation_live_run_card [data-testid="stMetric"] { min-height:76px; padding:10px 12px; border:1px solid #383d3b; border-radius:11px; background:#202221a8; }
.st-key-conversation_live_run_card [data-testid="stMetricLabel"] p { color:#a4aaa7; font-size:11px; }
.st-key-conversation_live_run_card [data-testid="stMetricValue"] { color:#e4ede9; font-size:1.05rem; }
.st-key-conversation_live_run_card [data-testid="stExpander"] { border-color:#414744; border-radius:11px; background:#20222180; }
.st-key-conversation_live_run_card [data-testid="stCaptionContainer"] p { line-height:1.55; }
[data-testid="stCaptionContainer"] { color:var(--chat-muted); }
[class*="st-key-client_question_timer_"] [data-testid="stHorizontalBlock"] {
  flex-direction:column!important; align-items:flex-end!important; gap:.2rem!important;
}
[class*="st-key-client_question_timer_"] [data-testid="stCaptionContainer"] p {
  text-align:right; margin-bottom:0;
}
[class*="st-key-client_question_timer_"] [data-testid="stButton"] { align-self:flex-end; }
button:focus-visible { outline:2px solid #b9b9b9; outline-offset:2px; }
.st-key-conversation_composer textarea:focus-visible { outline:none; }

/* First-visit guide: concise, diagram-led, and animated with reduced-motion support. */
.labcd-intro-kicker { display:flex; align-items:center; gap:9px; color:#aebaff; font:600 10px/1.4 ui-monospace,monospace; letter-spacing:.16em; text-transform:uppercase; }
.labcd-onboarding-mark, .labcd-onboarding-mark .labcd-icon { display:block; width:24px; height:24px; }
.labcd-intro-title { color:#f1f5f7; font-size:clamp(22px,3vw,31px); font-weight:600; letter-spacing:-.035em; line-height:1.18; margin:7px 0 6px; }
.labcd-intro-copy { color:#b7c0c9; font-size:14px; line-height:1.55; margin:0 0 10px; max-width:760px; }
.labcd-tour-progress { display:flex; gap:7px; margin:2px 0 8px; }
.labcd-tour-progress span { flex:1; height:4px; border-radius:99px; background:#343b45; }
.labcd-tour-progress span.done { background:linear-gradient(90deg,#54c9b0,#5ea8ea); box-shadow:0 0 12px #54c9b044; }
.labcd-tour-count { color:#9ba9b7; font:11px/1.4 ui-monospace,monospace; letter-spacing:.08em; margin:0 0 3px; }
.labcd-tour-visual { display:block; width:100%; height:auto; max-height:250px; margin:3px auto 9px; border:1px solid #303945; border-radius:16px; background:radial-gradient(ellipse at 50% 100%,#1d2d3b,#141920 72%); }
.labcd-svg-label { fill:#dce8f4; font:600 12px Arial,sans-serif; }
.labcd-svg-muted { fill:#8fa1b2; font:10px Arial,sans-serif; }
.labcd-svg-grid { fill:none; stroke:#344252; stroke-width:1; opacity:.7; }
.labcd-svg-card { fill:#19222d; stroke:#344456; stroke-width:1.2; }
.labcd-svg-blue { fill:#1d3652; stroke:#4c83ba; stroke-width:1.3; }
.labcd-svg-teal { fill:#183630; stroke:#4b9885; stroke-width:1.3; }
.labcd-svg-purple { fill:#282640; stroke:#7773bc; stroke-width:1.3; }
.labcd-svg-line { fill:none; stroke:#58cdb4; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
.labcd-svg-line-blue { fill:none; stroke:#6caef0; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
.labcd-svg-dash { fill:none; stroke:#79d8c6; stroke-width:1.8; stroke-dasharray:5 7; animation:labcd-dash-flow 2.4s linear infinite; }
.labcd-svg-forecast { fill:none; stroke:#b5a8ff; stroke-width:2.5; stroke-dasharray:5 5; animation:labcd-dash-flow 2.8s linear infinite; }
.labcd-svg-dot { fill:#baffee; filter:drop-shadow(0 0 6px #78f2d1); animation:labcd-dot-pulse 1.8s ease-in-out infinite; }
.labcd-svg-neuron { fill:#193b38; stroke:#59bca8; stroke-width:1.5; animation:labcd-neuron-glow 2.3s ease-in-out infinite; }
.labcd-tour-tip { display:flex; align-items:center; gap:8px; color:#bfd6d3; border:1px solid #304943; background:#172521; border-radius:10px; padding:9px 12px; font-size:12px; line-height:1.45; margin:1px 0 8px; }
@keyframes labcd-dash-flow { to { stroke-dashoffset:-36; } }
@keyframes labcd-dot-pulse { 0%,100% { opacity:.55; r:4; } 50% { opacity:1; r:7; } }
@keyframes labcd-neuron-glow { 0%,100% { filter:drop-shadow(0 0 1px #59bca844); } 50% { filter:drop-shadow(0 0 7px #59bca8cc); } }
@media(max-width:620px) { .labcd-tour-visual { max-height:180px; } .labcd-intro-copy { font-size:13px; } }
@media(prefers-reduced-motion:reduce) {
  .labcd-svg-dash,.labcd-svg-forecast,.labcd-svg-dot,.labcd-svg-neuron,
  .run-tour-prediction,.run-tour-wire,.run-tour-input-wave,.run-tour-memory,
  .run-tour-neuron--active,.run-learning-snapshot { animation:none!important; }
  .run-tour-svg animateMotion { display:none; }
}
@media(max-width:850px) {
  .stMainBlockContainer { padding:1rem 1rem 2rem; }
  [data-testid="stBottomBlockContainer"] { padding:0 1rem .5rem; }
  .conversation-welcome { padding-top:8vh; }
  .conversation-welcome h1 { font-size:27px; }
  .conversation-heading { gap:7px; font-size:12px; }
  .conversation-heading .labcd-logo { width:96px; }
  .run-process-flow { grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
  .run-process-arrow { display:none; }
  .run-search-block { grid-column:1/-1; }
  .st-key-conversation_file_panel {
    inset:0 0 0 auto!important; width:min(290px,86vw)!important; min-width:0!important;
    height:100dvh!important; max-height:none!important; border-radius:0;
  }
  .st-key-conversation_file_panel_expanded {
    inset:8px!important; width:auto!important; height:calc(100dvh - 16px)!important; max-height:none!important;
  }
}
@media(max-width:720px) {
  .run-tour-layout { grid-template-columns:minmax(0,1fr); gap:9px; }
  .run-tour-art { min-height:0; }
  .run-tour-svg { max-height:none; }
  .run-tour-story { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
  .run-tour-general,.run-tour-specific { padding:10px; }
}
@media(max-width:480px) {
  .run-learning-snapshot { padding:11px; }
  .run-tour-story { grid-template-columns:minmax(0,1fr); }
  .run-tour-general p,.run-tour-specific p { font-size:11px; }
  .run-tour-current-meta span { font-size:9px; }
}
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto!important; transition:none!important; animation:none!important; } }
</style>
"""
