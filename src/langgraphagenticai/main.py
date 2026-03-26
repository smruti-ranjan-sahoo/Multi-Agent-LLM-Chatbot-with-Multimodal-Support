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

# ── Fixed user ID (no auth) ───────────────────────────────────────────────
user_id = "local_user"

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
    return (
        '<div class="upload-card-header">'
        '<div class="upload-icon-box ' + icon_class + '">' + icon + '</div>'
        '<div>'
        '<div class="upload-card-title">' + title + '</div>'
        '<div class="upload-card-sub">' + sub + '</div>'
        '</div>'
        '</div>'
    )

def _success_pill(msg):
    return (
        '<div class="upload-success">'
        '<div class="success-dot"></div>'
        '<span>' + msg + '</span>'
        '</div>'
    )

def _transcript_html(text):
    return (
        '<div class="transcript-box">'
        '<div class="transcript-label">🎤 You said</div>'
        + text +
        '</div>'
    )

def _voice_card_header():
    return """
    <div class="voice-card">
        <div class="voice-card-row">
            <div class="voice-icon-box">🎙</div>
            <div>
                <div class="voice-card-title">Voice Input</div>
                <div class="voice-card-sub">Upload a WAV · transcribed automatically</div>
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

    # ── Session state defaults (MUST be inside function so they always run) ──
    for key, default in [
        ("last_audio_id",    None),
        ("last_audio_path",  None),
        ("voice_transcript", None),
        ("voice_sent",       False),
        ("image_bytes",      None),
        ("retriever",        None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    ui         = LoadStreamlitUI()
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
            st.markdown(
                '<div class="chat-wrap">'
                '<div class="chat-row user-row">'
                '<div class="chat-avatar usr-av">U</div>'
                '<div class="bubble user-bubble">' + user_message + '</div>'
                '</div>'
                '<div class="chat-row">'
                '<div class="chat-avatar ai-av">AI</div>'
                '<div class="bubble ai-bubble">' + ai_msg.content + '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
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

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("+ New Chat"):
                tid = str(uuid.uuid4())
                threads[tid] = {"title": "New Chat", "messages": []}
                uc_store["current_thread_id"] = tid
                current_tid = tid
                st.session_state.image_bytes      = None
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
                st.session_state.image_bytes      = None
                st.session_state.voice_transcript = None
                st.session_state.voice_sent       = False

        thread_ids = list(threads.keys())
        titles     = {tid: threads[tid]["title"] for tid in thread_ids}
        selected_title = st.selectbox(
            f"{usecase} Threads",
            list(titles.values()),
            index=list(thread_ids).index(current_tid),
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
    if usecase == "Document QA":
        st.markdown(
            '<div class="upload-card">'
            + _section_header("📄", "Document Upload", "PDF or CSV · used as RAG context", "doc-icon"),
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop your file here", type=["pdf", "csv"],
            label_visibility="collapsed", key=f"doc_upload_{current_tid}",
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
                unsafe_allow_html=True,
            )
        elif st.session_state.retriever:
            st.markdown(
                _success_pill("Document already loaded — ready to answer questions"),
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    retriever = st.session_state.retriever

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-AGENT IMAGE UPLOAD
    # ═══════════════════════════════════════════════════════════════════
    if usecase == "Multi-Agent Chatbot":
        st.markdown(
            '<div class="upload-card">'
            + _section_header("🖼️", "Image Upload", "Optional · PNG or JPG · Vision models only", "img-icon"),
            unsafe_allow_html=True,
        )
        uploaded_image = st.file_uploader(
            "Drop an image here", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key=f"img_upload_{current_tid}",
        )
        if uploaded_image:
            st.session_state.image_bytes = uploaded_image.read()

        if st.session_state.image_bytes:
            st.markdown(
                '<div class="img-preview-wrap">'
                '<div class="img-preview-label">Image attached to next message</div>',
                unsafe_allow_html=True,
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
        st.markdown(_voice_card_header(), unsafe_allow_html=True)

        col_rec, col_send, col_clear = st.columns([3, 3, 1])

        with col_rec:
            uploaded_audio = st.file_uploader(
                "Upload WAV / MP3", type=["wav", "mp3"],
                label_visibility="collapsed", key="voice_upload",
            )
            if uploaded_audio and not st.session_state.voice_sent:
                with st.spinner("🔄 Transcribing…"):
                    suffix = "." + uploaded_audio.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_audio.read())
                        tmp_path = tmp.name
                    st.session_state.voice_transcript = speech_to_text(tmp_path)
                    st.session_state.voice_sent       = False

        with col_send:
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

        if st.session_state.voice_transcript:
            st.markdown(
                _transcript_html(st.session_state.voice_transcript),
                unsafe_allow_html=True,
            )

    # ── Reset conversation ─────────────────────────────────────────────
    _, col_rst = st.columns([10, 1])
    with col_rst:
        if st.button("🗑", key="stateful_reset", help="Clear conversation"):
            messages.clear()
            st.session_state.image_bytes      = None
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
            image=st.session_state.image_bytes,
        ).display_result_on_ui()

        # Persist assistant reply
        if messages and isinstance(messages[-1], AIMessage):
            db.add_message(f"{user_id}_{current_tid}", "assistant", messages[-1].content)

        # TTS — only regenerate for new responses
        if messages and isinstance(messages[-1], AIMessage):
            latest_text      = messages[-1].content
            current_audio_id = hash(latest_text)

            # FIX: use .get() as a safe fallback in case key is somehow missing
            if st.session_state.get("last_audio_id") != current_audio_id:
                st.session_state.last_audio_path = text_to_speech(latest_text)
                st.session_state.last_audio_id   = current_audio_id

            if st.session_state.get("last_audio_path"):
                st.audio(st.session_state.last_audio_path)