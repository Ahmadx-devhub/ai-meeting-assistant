import streamlit as st
import time
import html as _html
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b0b12;
    --surface: #12121b;
    --surface-2: #1a1a26;
    --border: #26263a;
    --accent: #7c5cff;
    --accent-glow: #a78bfa;
    --accent-2: #22d3ee;
    --text: #e6e6f0;
    --text-muted: #8b8ba7;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(124,92,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,92,255,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none; z-index: 0;
}

h1,h2,h3,h4,h5,h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800; line-height: 1.05; margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 55%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.85rem; color: var(--text-muted);
    margin-top: 0.5rem; letter-spacing: 0.02em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text); }

.side-brand {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem;
    background: linear-gradient(135deg, #fff 0%, var(--accent-glow) 60%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.15rem;
}
.side-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase;
}

/* Bordered containers act as cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease;
    padding: 0.15rem 0;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(124,92,255,0.45) !important;
}

.card-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--accent-glow);
    margin-bottom: 0.75rem; padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.55rem;
}
.card-header .icon { font-size: 1rem; }

/* ── The fix: properly styled markdown prose ── */
[data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif; font-size: 0.92rem;
    line-height: 1.75; color: var(--text); margin: 0.5rem 0;
}
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol {
    padding-left: 1.4rem; margin: 0.6rem 0;
}
[data-testid="stMarkdownContainer"] li {
    font-family: 'Inter', sans-serif; font-size: 0.92rem;
    line-height: 1.7; color: var(--text); margin: 0.35rem 0;
}
[data-testid="stMarkdownContainer"] li::marker { color: var(--accent-glow); }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    font-family: 'Syne', sans-serif; color: var(--text);
    margin: 1rem 0 0.5rem; line-height: 1.3;
}
[data-testid="stMarkdownContainer"] h1 { font-size: 1.25rem; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.1rem;  }
[data-testid="stMarkdownContainer"] h3 { font-size: 1rem;    }
[data-testid="stMarkdownContainer"] h4 { font-size: 0.95rem; }
[data-testid="stMarkdownContainer"] strong { color: var(--accent-glow); font-weight: 600; }
[data-testid="stMarkdownContainer"] em     { color: var(--accent-2);    font-style: italic; }
[data-testid="stMarkdownContainer"] code {
    background: var(--surface-2); color: var(--accent-2);
    padding: 0.1rem 0.4rem; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
[data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid var(--accent); padding-left: 1rem;
    margin: 0.75rem 0; color: var(--text-muted); font-style: italic;
}

/* Badges */
.badge {
    display: inline-block; padding: 0.25rem 0.65rem; border-radius: 6px;
    font-size: 0.65rem; font-family: 'JetBrains Mono', monospace;
    font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;
}
.badge-purple { background: rgba(124,92,255,0.15); color: var(--accent-glow); border: 1px solid rgba(124,92,255,0.3); }
.badge-cyan   { background: rgba(34,211,238,0.12); color: var(--accent-2);    border: 1px solid rgba(34,211,238,0.3); }
.badge-green  { background: rgba(16,185,129,0.12); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* Inputs & buttons */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,92,255,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.2s !important; text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(124,92,255,0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Status bar */
.status-bar {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.55rem 0.8rem; background: var(--surface-2);
    border-radius: 8px; margin: 0.3rem 0;
    border: 1px solid var(--border);
    font-size: 0.75rem; font-family: 'Inter', sans-serif;
    color: var(--text-muted);
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done    { background: var(--success); }
.dot-pending { background: var(--border); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* Chat */
.chat-container {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem;
    max-height: 480px; overflow-y: auto; margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.3rem; }
.chat-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase;
}
.chat-bubble {
    display: inline-block; padding: 0.7rem 1rem; border-radius: 12px;
    font-family: 'Inter', sans-serif; font-size: 0.88rem;
    line-height: 1.65; max-width: 85%; word-wrap: break-word;
}
.user-label { color: var(--accent-glow); }
.bot-label  { color: var(--accent-2); }
.user-bubble { background: rgba(124,92,255,0.12); border: 1px solid rgba(124,92,255,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(34,211,238,0.08);  border: 1px solid rgba(34,211,238,0.2); align-self: flex-start; }

/* Transcript */
.transcript-box {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem;
    font-family: 'Inter', sans-serif; font-size: 0.85rem;
    line-height: 1.8; max-height: 340px; overflow-y: auto;
    color: var(--text-muted); white-space: pre-wrap; word-break: break-word;
}

/* Streamlit misc */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
label { color: var(--text-muted) !important; font-size: 0.82rem !important; font-family: 'Inter', sans-serif !important; }

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--text) !important;
}

/* Empty state */
.empty-state { display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 5rem 2rem; text-align: center; }
.empty-state .icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state .title { font-family: 'Syne', sans-serif; font-size: 1.5rem;
    font-weight: 700; color: var(--text); margin-bottom: 0.5rem; }
.empty-state .desc { color: var(--text-muted); font-size: 0.9rem;
    max-width: 420px; line-height: 1.7; }
.empty-state .badges { margin-top: 1.5rem; display: flex;
    gap: 0.6rem; flex-wrap: wrap; justify-content: center; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.25rem 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Tabs (main navigation) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: transparent;
    padding: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text-muted);
    background: transparent;
    padding: 0.75rem 1.25rem;
    border-radius: 8px 8px 0 0;
    border-bottom: 2px solid transparent;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text);
    background: rgba(124,92,255,0.05);
}
.stTabs [aria-selected="true"] {
    color: var(--accent-glow) !important;
    border-bottom-color: var(--accent) !important;
    background: rgba(124,92,255,0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Fix: long content (Open Questions) no longer breaks layout ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    min-width: 0 !important;
    overflow-wrap: anywhere;
    word-break: break-word;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] {
    max-width: 100%;
    overflow-x: auto;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] pre {
    overflow-x: auto;
    max-width: 100%;
}
[data-testid="column"] { min-width: 0 !important; }

</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "result": None, "chat_history": [], "processing": False,
    "pipeline_done": False, "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def esc(s): return _html.escape(str(s))

def render_card(icon: str, title: str, content: str):
    """Bordered container + real markdown rendering — this is the fix."""
    with st.container(border=True):
        st.markdown(
            f'<div class="card-header"><span class="icon">{icon}</span>{title}</div>',
            unsafe_allow_html=True,
        )
        if content and str(content).strip():
            st.markdown(content)   # ← real markdown parsing
        else:
            st.markdown(
                '<p style="color:var(--text-muted);font-style:italic">No content extracted.</p>',
                unsafe_allow_html=True,
            )

def step_status(steps, key):
    s = steps.get(key, "pending")
    if s == "active": return "dot-active"
    if s == "done":   return "dot-done"
    return "dot-pending"

def render_step_bar(label, key, icon):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(
        f'<div class="status-bar"><div class="status-dot {css}"></div>'
        f'<span>{icon} {label}</span></div>',
        unsafe_allow_html=True,
    )

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="side-brand">🎬 AI Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input(
        "Source", placeholder="YouTube URL or file path",
        label_visibility="collapsed",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    # Show pipeline status during AND after processing
    if st.session_state.pipeline_steps:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio", "🔊", "Audio"),
            ("transcript", "📝", "Transcription"),
            ("title", "🏷️", "Title"),
            ("summary", "📋", "Summary"),
            ("extract", "🔍", "Extraction"),
            ("rag", "🧠", "RAG"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        def set_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with st.status("⚙️ Running pipeline…", expanded=True) as status:
                st.write("🔊 Processing audio…")
                set_step("audio", "active")
                chunks = process_input(source)
                set_step("audio", "done")

                st.write("📝 Transcribing…")
                set_step("transcript", "active")
                transcript = transcribe_all(chunks, language)
                set_step("transcript", "done")

                st.write("🏷️ Generating title…")
                set_step("title", "active")
                title = generate_title(transcript)
                set_step("title", "done")

                st.write("📋 Summarising…")
                set_step("summary", "active")
                summary = summarize(transcript)
                set_step("summary", "done")

                st.write("🔍 Extracting action items, decisions, questions…")
                set_step("extract", "active")
                action_items = extract_action_items(transcript)
                decisions    = extract_key_decisions(transcript)
                questions    = extract_questions(transcript)
                set_step("extract", "done")

                st.write("🧠 Building RAG engine…")
                set_step("rag", "active")
                rag_chain = build_rag_chain(transcript)
                set_step("rag", "done")

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "action_items": action_items, "key_decisions": decisions,
                "open_questions": questions, "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            st.error(f"❌ Error: {e}")
# ─── Results ────────────────────────────────────────────────────────────────────
# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # ── Title banner ──
    with st.container(border=True):
        st.markdown(
            '<div class="card-header"><span class="icon">📌</span>Session Title</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family:\'Syne\',sans-serif;font-size:1.35rem;'
            f'font-weight:700;line-height:1.35;color:var(--text)">{esc(r["title"])}</div>',
            unsafe_allow_html=True,
        )

    # ── Two parallel views: Analysis | Chat ──
    tab_analysis, tab_chat = st.tabs(["📊   Analysis", "💬   Chat with Meeting"])

    # ═════════════════════════════ ANALYSIS TAB ═════════════════════════════
    with tab_analysis:
        render_card("📋", "Summary", r["summary"])

        with st.expander("📝   Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{esc(r["transcript"])}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        # Sub-tabs give each extractor FULL WIDTH → no more layout breakage
        sub_a, sub_b, sub_c = st.tabs(
            ["✅   Action Items", "🔑   Key Decisions", "❓   Open Questions"]
        )
        with sub_a:
            render_card("✅", "Action Items", r["action_items"])
        with sub_b:
            render_card("🔑", "Key Decisions", r["key_decisions"])
        with sub_c:
            render_card("❓", "Open Questions", r["open_questions"])

    # ═════════════════════════════ CHAT TAB ═════════════════════════════
    with tab_chat:
        st.markdown(
            '<div style="font-family:\'Syne\',sans-serif;font-size:1.15rem;'
            'font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>',
            unsafe_allow_html=True,
        )

        # Chat history
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                safe = esc(msg["content"]).replace("\n", "<br>")
                if msg["role"] == "user":
                    chat_html += (
                        '<div class="chat-msg" style="align-items:flex-end">'
                        '<span class="chat-label user-label">You</span>'
                        f'<div class="chat-bubble user-bubble">{safe}</div></div>'
                    )
                else:
                    chat_html += (
                        '<div class="chat-msg" style="align-items:flex-start">'
                        '<span class="chat-label bot-label">🤖 Assistant</span>'
                        f'<div class="chat-bubble bot-bubble">{safe}</div></div>'
                    )
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:var(--surface);border:1px dashed var(--border);'
                'border-radius:14px;padding:3rem 2rem;text-align:center">'
                '<div style="font-size:2.25rem;margin-bottom:0.75rem">💬</div>'
                '<div style="color:var(--text-muted);font-size:0.92rem;line-height:1.6">'
                "Ask anything about your meeting transcript.<br>"
                "Try: <em style=\"color:var(--accent-2)\">What were the main decisions?</em>"
                "</div></div>",
                unsafe_allow_html=True,
            )

        # Chat input (form → auto-clears after send)
        with st.form("chat_form", clear_on_submit=True):
            chat_col1, chat_col2 = st.columns([5, 1], gap="small")
            with chat_col1:
                user_input = st.text_input(
                    "Your question",
                    placeholder="Ask anything about the meeting…",
                    label_visibility="collapsed",
                )
            with chat_col2:
                send_btn = st.form_submit_button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary", key="chat_clear"):
                st.session_state.chat_history = []
                st.rerun()

else:
    # ── Empty state (unchanged) ──
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎬</div>
        <div class="title">Ready to Analyse</div>
        <div class="desc">Paste a YouTube URL or local file path in the sidebar,
        choose your language, and hit <strong style="color:var(--accent-glow)">Analyse</strong>
        to get started.</div>
        <div class="badges">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)