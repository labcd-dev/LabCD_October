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
[data-testid="stCaptionContainer"] { color:var(--chat-muted); }
button:focus-visible { outline:2px solid #b9b9b9; outline-offset:2px; }
.st-key-conversation_composer textarea:focus-visible { outline:none; }
@media(max-width:850px) {
  .stMainBlockContainer { padding:1rem 1rem 2rem; }
  [data-testid="stBottomBlockContainer"] { padding:0 1rem .5rem; }
  .conversation-welcome { padding-top:8vh; }
  .conversation-welcome h1 { font-size:27px; }
  .conversation-heading { font-size:12px; }
  .st-key-conversation_file_panel {
    top:auto; right:12px; bottom:102px; left:12px; width:auto;
    max-height:55dvh;
  }
  .st-key-conversation_file_panel [data-testid="stVerticalBlockBorderWrapper"] { max-height:55dvh; }
}
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto!important; transition:none!important; animation:none!important; } }
</style>
"""
