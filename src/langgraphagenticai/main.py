from distro import name
import streamlit as st
import uuid
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

from gtts import gTTS
from scipy.io.wavfile import write
from src.langgraphagenticai.voice_utils import speech_to_text
from src.langgraphagenticai.db.database import ChatDatabase
from src.langgraphagenticai.rag.document_loader import load_and_chunk
from src.langgraphagenticai.rag.vector_store import create_vector_store
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit
from src.langgraphagenticai.auth import supabase, login_google, login_github
import streamlit.components.v1 as components
LOGIN_BG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
 
/* ── Full bleed dark canvas ── */
section[data-testid="stMain"] > div:first-child,
.stApp { background: #060810 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"],
footer { display: none !important; }
 
/* ── Neural grid ── */
.login-bg {
    position: fixed; inset: 0; background: #060810; z-index: 0; overflow: hidden;
    pointer-events: none;
}
.login-bg::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(79,126,247,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,126,247,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridDrift 25s linear infinite;
}
@keyframes gridDrift { 0%{transform:translate(0,0)} 100%{transform:translate(60px,60px)} }
 
/* ── Ambient orbs ── */
.orb { position:absolute; border-radius:50%; filter:blur(90px); animation:orbFloat linear infinite; }
.orb-1 { width:600px;height:600px; background:rgba(79,126,247,0.10); top:-200px;left:-200px; animation-duration:20s; }
.orb-2 { width:500px;height:500px; background:rgba(124,79,224,0.09); bottom:-100px;right:-150px; animation-duration:26s;animation-delay:-8s; }
.orb-3 { width:350px;height:350px; background:rgba(46,204,113,0.06); top:35%;left:55%; animation-duration:18s;animation-delay:-4s; }
@keyframes orbFloat {
    0%,100% { transform:translate(0,0) scale(1); }
    33%      { transform:translate(40px,-50px) scale(1.06); }
    66%      { transform:translate(-30px,30px) scale(0.94); }
}
 
/* ── Floating chat bubbles ── */
.bg-bubble {
    position:absolute;
    background:rgba(255,255,255,0.018);
    border:1px solid rgba(79,126,247,0.07);
    border-radius:14px; padding:9px 15px;
    font-family:'JetBrains Mono',monospace; font-size:11px;
    color:rgba(180,200,255,0.13); white-space:nowrap;
    animation:bubbleDrift linear infinite; pointer-events:none;
    backdrop-filter:blur(4px);
}
@keyframes bubbleDrift {
    0%   { transform:translateY(110vh) translateX(0); opacity:0; }
    4%   { opacity:1; }
    96%  { opacity:1; }
    100% { transform:translateY(-180px) translateX(var(--drift,0px)); opacity:0; }
}
 
/* ── Typing indicators ── */
.bg-typing {
    position:absolute; display:flex; gap:4px; align-items:center;
    padding:11px 15px;
    background:rgba(255,255,255,0.012);
    border:1px solid rgba(79,126,247,0.05);
    border-radius:12px; animation:bubbleDrift linear infinite; pointer-events:none;
}
.bg-typing span {
    width:5px;height:5px;border-radius:50%; background:rgba(79,126,247,0.28);
    animation:tdot 1.3s ease-in-out infinite;
}
.bg-typing span:nth-child(2){animation-delay:.2s;background:rgba(100,100,220,0.28);}
.bg-typing span:nth-child(3){animation-delay:.4s;background:rgba(124,79,224,0.28);}
@keyframes tdot { 0%,80%,100%{transform:translateY(0) scale(0.8);opacity:.4;} 40%{transform:translateY(-6px) scale(1);opacity:1;} }
 
/* ── Canvas ── */
#neural-canvas { position:fixed;inset:0;z-index:1;pointer-events:none;opacity:0.35; }
 
/* ── Login card wrapper ── */
.login-outer {
    position: relative; z-index: 10;
    min-height: 100vh;
    display: flex;
    align-items: center;
    padding: 40px 60px;
}
 
/* ── Card ── */
.login-card {
    width: 400px;
    background: rgba(10,13,24,0.90);
    border: 1px solid rgba(79,126,247,0.16);
    border-radius: 24px;
    padding: 44px 38px 38px;
    backdrop-filter: blur(50px);
    box-shadow:
        0 40px 80px rgba(0,0,0,0.65),
        0 0 0 1px rgba(79,126,247,0.04),
        0 0 140px rgba(79,126,247,0.07);
    animation: cardIn .7s cubic-bezier(.16,1,.3,1) both;
}
@keyframes cardIn { from{opacity:0;transform:translateY(28px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }
 
.login-logo {
    width:52px;height:52px;
    background:linear-gradient(135deg,#4F7EF7,#7C4FE0);
    border-radius:14px;
    display:flex;align-items:center;justify-content:center;
    font-size:18px;font-weight:700;color:#fff;
    margin:0 auto 22px;
    box-shadow:0 8px 28px rgba(79,126,247,0.35);
    animation:logoPulse 3s ease-in-out infinite;
}
@keyframes logoPulse {
    0%,100%{box-shadow:0 8px 28px rgba(79,126,247,.35);}
    50%     {box-shadow:0 8px 36px rgba(79,126,247,.55),0 0 60px rgba(79,126,247,.12);}
}
.login-title {
    font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700;
    color:#E8EAF6; text-align:center; margin-bottom:6px;
}
.login-sub {
    font-size:12.5px; color:#40455E; text-align:center;
    margin-bottom:30px; line-height:1.55;
}
.login-divider {
    display:flex; align-items:center; gap:12px;
    color:#1E2235; font-size:10px; margin:18px 0;
}
.login-divider::before,.login-divider::after { content:''; flex:1; height:1px; background:#151825; }
 
.feature-row {
    display:flex; gap:6px; justify-content:center;
    margin-top:26px; flex-wrap:wrap;
}
.feat-tag {
    font-size:10px; color:#2A3050;
    background:rgba(255,255,255,0.02);
    border:1px solid #151825; border-radius:20px; padding:4px 10px;
}
 
/* Streamlit button overrides within login */
.login-card .stButton > button {
    width:100% !important;
    border-radius:11px !important;
    padding:12px 16px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-size:13.5px !important; font-weight:500 !important;
    transition: all .2s !important;
    letter-spacing:.2px !important;
}
.login-card .stButton:first-of-type > button {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    color:#B0B8D8 !important;
}
.login-card .stButton:first-of-type > button:hover {
    background:rgba(255,255,255,0.07) !important;
    border-color:rgba(255,255,255,0.16) !important;
    transform:translateY(-1px) !important;
}
.login-card .stButton:last-of-type > button {
    background:rgba(79,126,247,0.07) !important;
    border:1px solid rgba(79,126,247,0.2) !important;
    color:#7AA0F0 !important;
}
.login-card .stButton:last-of-type > button:hover {
    background:rgba(79,126,247,0.13) !important;
    border-color:rgba(79,126,247,0.35) !important;
    transform:translateY(-1px) !important;
}
h1 { display:none !important; }
</style>
"""
 
LOGIN_BG_HTML = """
<!-- Animated neural background -->
<div class="login-bg">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
 
    <div class="bg-bubble" style="left:4%;  --drift:20px;  animation-duration:14s; animation-delay:0s;">  Analyze this dataset for anomalies...</div>
    <div class="bg-bubble" style="left:12%; --drift:-25px; animation-duration:19s; animation-delay:-5s;"> I found 3 outliers in columns B and F.</div>
    <div class="bg-bubble" style="left:68%; --drift:15px;  animation-duration:16s; animation-delay:-3s;"> Write a Python function for binary search</div>
    <div class="bg-bubble" style="left:78%; --drift:-20px; animation-duration:21s; animation-delay:-9s;"> def binary_search(arr, target): ...</div>
    <div class="bg-bubble" style="left:48%; --drift:28px;  animation-duration:13s; animation-delay:-7s;"> Summarize this 50-page PDF</div>
    <div class="bg-bubble" style="left:33%; --drift:-15px; animation-duration:23s; animation-delay:-11s;">Translate to French with formal tone</div>
    <div class="bg-bubble" style="left:86%; --drift:10px;  animation-duration:17s; animation-delay:-2s;"> Generate a SQL query for monthly revenue</div>
    <div class="bg-bubble" style="left:58%; --drift:-25px; animation-duration:20s; animation-delay:-14s;">Explain neural networks simply</div>
    <div class="bg-bubble" style="left:22%; --drift:18px;  animation-duration:15s; animation-delay:-6s;"> What are the key risks in this contract?</div>
    <div class="bg-bubble" style="left:75%; --drift:-12px; animation-duration:18s; animation-delay:-16s;">Create a marketing strategy for Q4</div>
 
    <div class="bg-typing" style="left:20%; --drift:20px;  animation-duration:12s; animation-delay:-4s;"><span></span><span></span><span></span></div>
    <div class="bg-typing" style="left:72%; --drift:-10px; animation-duration:15s; animation-delay:-10s;"><span></span><span></span><span></span></div>
    <div class="bg-typing" style="left:44%; --drift:14px;  animation-duration:11s; animation-delay:-1s;"><span></span><span></span><span></span></div>
    <div class="bg-typing" style="left:90%; --drift:-18px; animation-duration:14s; animation-delay:-8s;"><span></span><span></span><span></span></div>
</div>
 
<canvas id="neural-canvas"></canvas>
<script>
(function(){
    const c = document.getElementById('neural-canvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    function resize(){ c.width=window.innerWidth; c.height=window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);
 
    const N = 50;
    const nodes = Array.from({length:N}, ()=>({
        x: Math.random()*c.width,
        y: Math.random()*c.height,
        vx:(Math.random()-.5)*.35,
        vy:(Math.random()-.5)*.35,
        r: Math.random()*1.8+.8,
        ph: Math.random()*Math.PI*2
    }));
 
    // Mouse influence
    let mx=-1000, my=-1000;
    window.addEventListener('mousemove', e=>{ mx=e.clientX; my=e.clientY; });
 
    function draw(){
        ctx.clearRect(0,0,c.width,c.height);
        const now = Date.now()/1000;
 
        nodes.forEach(n=>{
            n.ph += .015;
            // Gentle mouse repulsion
            const dx=n.x-mx, dy=n.y-my, d=Math.hypot(dx,dy);
            if(d<120){ n.vx+=dx/d*.08; n.vy+=dy/d*.08; }
            // Speed limit
            const speed=Math.hypot(n.vx,n.vy);
            if(speed>.8){ n.vx=(n.vx/speed)*.8; n.vy=(n.vy/speed)*.8; }
 
            n.x+=n.vx; n.y+=n.vy;
            if(n.x<0||n.x>c.width)  n.vx*=-1;
            if(n.y<0||n.y>c.height) n.vy*=-1;
 
            const a = .25+.15*Math.sin(n.ph);
            ctx.beginPath();
            ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
            ctx.fillStyle=`rgba(79,126,247,${a})`;
            ctx.fill();
        });
 
        // Connections
        for(let i=0;i<N;i++){
            for(let j=i+1;j<N;j++){
                const d=Math.hypot(nodes[i].x-nodes[j].x, nodes[i].y-nodes[j].y);
                if(d<140){
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x,nodes[i].y);
                    ctx.lineTo(nodes[j].x,nodes[j].y);
                    const a=.14*(1-d/140);
                    ctx.strokeStyle=`rgba(79,126,247,${a})`;
                    ctx.lineWidth=.6;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
"""

# ── Init session state ────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

if "auth_processed" not in st.session_state:
    st.session_state.auth_processed = False

if "logged_out" not in st.session_state:
    st.session_state.logged_out = False

query_params = st.query_params

# ── Handle OAuth redirect ─────────────────────────────────────────────────
if "code" in query_params and not st.session_state.auth_processed and not st.session_state.logged_out:
    code = query_params["code"]
    try:
        session = supabase.auth.exchange_code_for_session({"auth_code": code})
        user = supabase.auth.get_user()
        if user:
            st.session_state.user = user
            st.session_state.auth_processed = True
            st.session_state.logged_out = False
        st.query_params.clear()
    except Exception as e:
        st.error(f"Auth error: {e}")
        st.query_params.clear()

elif "code" in query_params and st.session_state.logged_out:
    st.query_params.clear()

# ── Login UI ──────────────────────────────────────────────────────────────
# ── Login UI ──────────────────────────────────────────────────────────────
if not st.session_state.user:
    st.markdown(LOGIN_BG_CSS, unsafe_allow_html=True)
    st.markdown(LOGIN_BG_HTML, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-outer">
        <div class="login-card">
            <div class="login-logo">LG</div>
            <div class="login-title">Welcome Back</div>
            <div class="login-sub">
                Sign in to your LangGraph Agentic AI workspace.<br>
                Multi-agent &middot; RAG &middot; Vision &middot; Voice
            </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([0.5, 3, 0.5])
    with col_c:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🔵  Continue with Google", key="google_btn"):
            url = login_google()
            st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("⬛  Continue with GitHub", key="github_btn"):
            url = login_github()
            st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">',
                        unsafe_allow_html=True)

    st.markdown("""
            <div class="login-divider">secure oauth</div>
            <div class="feature-row">
                <span class="feat-tag">🧠 Multi-Agent</span>
                <span class="feat-tag">📄 RAG</span>
                <span class="feat-tag">🖼 Vision</span>
                <span class="feat-tag">🎙 Voice</span>
                <span class="feat-tag">💾 Memory</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ── User available ────────────────────────────────────────────────────────
user    = st.session_state.user
user_id = user.user.id

# ── Session state defaults ────────────────────────────────────────────────
for key, default in [
    ("last_audio_id",      None),
    ("last_audio_path",    None),
    ("voice_transcript",   None),   # stores the STT result
    ("voice_sent",         False),  # True once Send is clicked
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═════════════════════════════════════════════════════════════════════════
# CSS — upload cards + voice card
# ═════════════════════════════════════════════════════════════════════════
UPLOAD_CSS = """
<style>
/* ── Upload zone card ── */
.upload-card {
    background: #12151F;
    border: 1px dashed #2A2F4A;
    border-radius: 14px;
    padding: 20px 18px 14px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #4F7EF7; }
.upload-card-header { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.upload-icon-box {
    width:36px; height:36px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}
.doc-icon { background:linear-gradient(135deg,#1A3A5C,#0F2440); border:1px solid #1E3A60; }
.img-icon { background:linear-gradient(135deg,#2A1A4A,#1A0F38); border:1px solid #3A1E60; }
.upload-card-title { font-size:14px; font-weight:600; color:#C8CCDF; }
.upload-card-sub   { font-size:11px; color:#3E4460; margin-top:2px; }

/* ── Success pill ── */
.upload-success {
    display:flex; align-items:center; gap:8px;
    background:#081a0e; border:1px solid #0f3a20;
    border-radius:8px; padding:8px 12px;
    font-size:12px; color:#2ECC71; margin-top:10px;
}
.success-dot {
    width:7px; height:7px; border-radius:50%;
    background:#2ECC71; flex-shrink:0;
    animation:blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Image preview ── */
.img-preview-wrap {
    background:#0E1019; border:1px solid #1C2035;
    border-radius:10px; padding:10px; margin-top:10px; text-align:center;
}
.img-preview-label {
    font-size:10px; color:#3E4460;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;
}

/* ── Voice card ── */
.voice-card {
    background:#12151F;
    border:1px solid #1C2035;
    border-radius:14px;
    padding:16px 18px 4px;
    margin-bottom:12px;
}
.voice-card-row {
    display:flex; align-items:center; gap:10px; margin-bottom:12px;
}
.voice-icon-box {
    width:34px; height:34px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:15px; flex-shrink:0;
    background:linear-gradient(135deg,#0F2A1A,#081A10);
    border:1px solid #1E5A1E;
}
.voice-card-title { font-size:13px; font-weight:600; color:#C8CCDF; }
.voice-card-sub   { font-size:11px; color:#3E4460; margin-top:1px; }

/* ── Transcript bubble ── */
.transcript-box {
    background:#0B0F1A;
    border:1px solid #1C2035;
    border-left:3px solid #4F7EF7;
    border-radius:0 10px 10px 0;
    padding:12px 16px;
    margin:10px 0 14px;
    font-size:13px; color:#B0BFEF;
    line-height:1.65; white-space:pre-wrap; word-break:break-word;
}
.transcript-label {
    font-size:10px; font-weight:700;
    letter-spacing:1.1px; text-transform:uppercase;
    color:#4F7EF7; margin-bottom:6px;
}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background:transparent !important;
    border:1px solid #2A1A3A !important;
    color:#7A4FBF !important;
    font-size:11px !important; border-radius:7px !important;
    padding:4px 12px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color:#7C4FE0 !important; color:#A77FE8 !important;
}
</style>
"""

# ── HTML helpers ──────────────────────────────────────────────────────────
def _section_header(icon, title, sub, icon_class="doc-icon"):
    return f"""
    <div class="upload-card-header">
        <div class="upload-icon-box {icon_class}">{icon}</div>
        <div>
            <div class="upload-card-title">{title}</div>
            <div class="upload-card-sub">{sub}</div>
        </div>
    </div>"""

def _success_pill(msg):
    return f"""
    <div class="upload-success">
        <div class="success-dot"></div>
        <span>{msg}</span>
    </div>"""

def _transcript_html(text):
    return f"""
    <div class="transcript-box">
        <div class="transcript-label">🎤 You said</div>
        {text}
    </div>"""

def _voice_card_header():
    return """
    <div class="voice-card">
        <div class="voice-card-row">
            <div class="voice-icon-box">🎙</div>
            <div>
                <div class="voice-card-title">Voice Input</div>
                <div class="voice-card-sub">Records 5 seconds · transcribed automatically</div>
            </div>
        </div>
    </div>"""

# ── TTS ───────────────────────────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    file_path = f"response_{uuid.uuid4()}.mp3"
    gTTS(text).save(file_path)
    return file_path

# ── DB ────────────────────────────────────────────────────────────────────
db = ChatDatabase()


# ═════════════════════════════════════════════════════════════════════════
def load_langgraph_agenticai_app():

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()
    if not user_input:
        return

    usecase             = user_input.get("selected_usecase")
    selected_llm        = user_input.get("selected_llm")
    selected_groq_model = user_input.get("selected_groq_model")

    st.markdown(UPLOAD_CSS, unsafe_allow_html=True)

    llm = GroqLLM(user_contols_input=user_input).get_llm_model()
    if not llm:
        st.error("LLM could not be initialized")
        return

    # ═══════════════════════════════════════════════════════════════════
    # BASIC CHATBOT
    # ═══════════════════════════════════════════════════════════════════
    if usecase == "Basic Chatbot":
        st.info("Basic Chatbot — Stateless Mode")
        user_message = st.chat_input("Type your message...")
        col_reset, _ = st.columns([1, 6])
        with col_reset:
            if st.button("🗑 Clear", key="basic_reset"):
                st.rerun()

        if user_message:
            graph  = GraphBuilder(llm).setup_graph(usecase)
            result = graph.invoke({"messages": [HumanMessage(content=user_message)]})
            ai_msg = result["messages"][-1]
            st.markdown(f"""
            <div class="chat-wrap">
              <div class="chat-row user-row">
                <div class="chat-avatar usr-av">U</div>
                <div class="bubble user-bubble">{user_message}</div>
              </div>
              <div class="chat-row">
                <div class="chat-avatar ai-av">AI</div>
                <div class="bubble ai-bubble">{ai_msg.content}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        return

    # ═══════════════════════════════════════════════════════════════════
    # STATEFUL MODES
    # ═══════════════════════════════════════════════════════════════════
    if "user_conversations" not in st.session_state:
        st.session_state.user_conversations = {}
    if user_id not in st.session_state.user_conversations:
        st.session_state.user_conversations[user_id] = {}
    if usecase not in st.session_state.user_conversations[user_id]:
        tid = str(uuid.uuid4())
        st.session_state.user_conversations[user_id][usecase] = {
            "current_thread_id": tid,
            "threads": {tid: {"title": f"{usecase} chat", "messages": []}}
        }
        db.create_conversation(f"{user_id}_{tid}")

    uc_store    = st.session_state.user_conversations[user_id][usecase]
    threads     = uc_store["threads"]
    current_tid = uc_store["current_thread_id"]

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.caption(f"👤 {user_id[:8]}…")

        if st.button("🚪 Logout"):
            try:
                supabase.auth.sign_out()
            except:
                pass
            st.session_state.clear()
            st.session_state.user          = None
            st.session_state.auth_processed = False
            st.session_state.logged_out    = True
            components.html("""
                <script>
                    Object.keys(localStorage).forEach(function(k) {
                        if (k.startsWith('sb-')) localStorage.removeItem(k);
                    });
                </script>
            """, height=0)
            st.rerun()

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("+ New Chat"):
                tid = str(uuid.uuid4())
                threads[tid] = {"title": "New Chat", "messages": []}
                uc_store["current_thread_id"] = tid
                current_tid = tid
                st.session_state.image_bytes     = None
                st.session_state.voice_transcript = None
                st.session_state.voice_sent       = False
                db.create_conversation(f"{user_id}_{tid}")
        with c2:
            if len(threads) > 1 and st.button("🗑 Delete"):
                db.delete_conversation(f"{user_id}_{current_tid}")
                del threads[current_tid]
                new_tid = list(threads.keys())[0]
                uc_store["current_thread_id"] = new_tid
                current_tid = new_tid
                st.session_state.image_bytes     = None
                st.session_state.voice_transcript = None
                st.session_state.voice_sent       = False

        thread_ids = list(threads.keys())
        titles     = {tid: threads[tid]["title"] for tid in thread_ids}
        selected_title = st.selectbox(
            f"{usecase} Threads",
            list(titles.values()),
            index=list(thread_ids).index(current_tid)
        )
        for tid, title in titles.items():
            if title == selected_title:
                uc_store["current_thread_id"] = tid
                current_tid = tid
                break

    thread   = threads[current_tid]
    messages = thread["messages"]

    if not messages:
        for role, content in db.get_messages(f"{user_id}_{current_tid}"):
            messages.append(
                HumanMessage(content=content) if role == "user" else AIMessage(content=content)
            )

    # ═══════════════════════════════════════════════════════════════════
    # DOCUMENT QA
    # ═══════════════════════════════════════════════════════════════════
    retriever = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if usecase == "Document QA":
        st.markdown(
            '<div class="upload-card">'
            + _section_header("📄", "Document Upload", "PDF or CSV · used as RAG context", "doc-icon"),
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Drop your file here", type=["pdf", "csv"],
            label_visibility="collapsed", key=f"doc_upload_{current_tid}"
        )
        if uploaded_file:
            st.session_state.retriever = None
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name
            with st.spinner("⚙️ Processing document…"):
                chunks      = load_and_chunk(file_path)
                vectorstore = create_vector_store(chunks, f"{user_id}_{current_tid}")
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.markdown(
                _success_pill(f"✓ {uploaded_file.name} loaded — {len(chunks)} chunks indexed"),
                unsafe_allow_html=True
            )
        elif st.session_state.retriever:
            st.markdown(
                _success_pill("Document already loaded — ready to answer questions"),
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    retriever = st.session_state.retriever

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-AGENT IMAGE UPLOAD
    # ═══════════════════════════════════════════════════════════════════
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None

    if usecase == "Multi-Agent Chatbot":
        st.markdown(
            '<div class="upload-card">'
            + _section_header("🖼️", "Image Upload", "Optional · PNG or JPG · Vision models only", "img-icon"),
            unsafe_allow_html=True
        )
        uploaded_image = st.file_uploader(
            "Drop an image here", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key=f"img_upload_{current_tid}"
        )
        if uploaded_image:
            st.session_state.image_bytes = uploaded_image.read()

        if st.session_state.image_bytes:
            st.markdown(
                '<div class="img-preview-wrap">'
                '<div class="img-preview-label">Image attached to next message</div>',
                unsafe_allow_html=True
            )
            st.image(st.session_state.image_bytes, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("✕ Remove image", key="clear_img", type="secondary"):
                st.session_state.image_bytes = None
                st.rerun()

            def _is_vision_model(n: str) -> bool:
                return any(k in (n or "").lower() for k in ("llama-4-scout", "vision-preview"))

            if selected_llm == "Groq" and not _is_vision_model(selected_groq_model):
                st.warning(
                    "⚠️ Vision model required for image input.\n\n"
                    "Switch to **llama-4-scout** or **llama-3.2-11b-vision-preview** in the sidebar."
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # VOICE / TEXT INPUT
    # ═══════════════════════════════════════════════════════════════════
    voice_mode   = st.toggle("🎙 Voice Mode", value=False)
    user_message = None

    # ── TEXT mode ─────────────────────────────────────────────────────
    if not voice_mode:
        st.session_state.voice_transcript = None
        st.session_state.voice_sent       = False
        user_message = st.chat_input("Type your message…")

    # ── VOICE mode ────────────────────────────────────────────────────
    else:
        # Header card (purely decorative — no Streamlit widgets inside)
        st.markdown(_voice_card_header(), unsafe_allow_html=True)

        # ── Row 1: Record · Send · Clear ──────────────────────────────
        col_rec, col_send, col_clear = st.columns([3, 3, 1])

        with col_rec:
            if st.button("🎤  Record  (5 s)", use_container_width=True, key="record_btn"):
                st.session_state.voice_sent = False
                with st.spinner("🔴  Recording… speak now"):
                    fs        = 44100
                    uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3"])

                    if uploaded_audio:
                        with open("temp.wav", "wb") as f:
                            f.write(uploaded_audio.read())

                        user_message = speech_to_text("temp.wav")
                    # recording = sd.rec(int(5 * fs), samplerate=fs, channels=1)
                    # sd.wait()
                    # tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    # write(tmp_wav.name, fs, recording)
                    # st.session_state.voice_transcript = speech_to_text(tmp_wav.name)

        with col_send:
            # Send button only appears when transcript is ready and not yet sent
            if st.session_state.voice_transcript and not st.session_state.voice_sent:
                if st.button("➤  Send to AI", use_container_width=True,
                             key="send_voice_btn", type="primary"):
                    user_message = st.session_state.voice_transcript
                    st.session_state.voice_sent = True

        with col_clear:
            if st.session_state.voice_transcript:
                if st.button("✕", key="clear_transcript", help="Discard recording"):
                    st.session_state.voice_transcript = None
                    st.session_state.voice_sent       = False
                    st.rerun()

        # ── Transcript bubble (shows what was recorded) ───────────────
        if st.session_state.voice_transcript:
            st.markdown(
                _transcript_html(st.session_state.voice_transcript),
                unsafe_allow_html=True
            )

    # ── Reset conversation ─────────────────────────────────────────────
    _, col_rst = st.columns([10, 1])
    with col_rst:
        if st.button("🗑", key="stateful_reset", help="Clear conversation"):
            messages.clear()
            st.session_state.image_bytes     = None
            st.session_state.voice_transcript = None
            st.session_state.voice_sent       = False
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # PROCESS MESSAGE
    # ═══════════════════════════════════════════════════════════════════
    if user_message:
        messages.append(HumanMessage(content=user_message))
        db.add_message(f"{user_id}_{current_tid}", "user", user_message)

        if len(messages) == 1:
            thread["title"] = user_message.strip()[:40]

        tools = {"search": lambda q: "Mock search result"}
        graph = GraphBuilder(llm=llm, tools=tools, retriever=retriever).setup_graph(usecase)

        DisplayResultStreamlit(
            usecase=usecase,
            graph=graph,
            messages=messages,
            image=st.session_state.image_bytes
        ).display_result_on_ui()

        # Persist assistant reply
        if messages and isinstance(messages[-1], AIMessage):
            db.add_message(f"{user_id}_{current_tid}", "assistant", messages[-1].content)

        # TTS — only regenerate for new responses
        if messages and isinstance(messages[-1], AIMessage):
            latest_text      = messages[-1].content
            current_audio_id = hash(latest_text)

            if st.session_state.last_audio_id != current_audio_id:
                st.session_state.last_audio_path = text_to_speech(latest_text)
                st.session_state.last_audio_id   = current_audio_id

            if st.session_state.last_audio_path:
                st.audio(st.session_state.last_audio_path)