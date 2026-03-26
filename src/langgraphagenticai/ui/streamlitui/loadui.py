# import streamlit as st
# from src.langgraphagenticai.ui.uiconfigfile import Config
# import uuid


# class LoadStreamlitUI:
#     def __init__(self):
#         self.config = Config()
#         self.user_controls = {}

#     def ensure_session_defaults(self):
#         if "selected_usecase" not in st.session_state:
#             st.session_state.selected_usecase = "Advanced Chatbot"
#         if "prev_usecase" not in st.session_state:
#             st.session_state.prev_usecase = st.session_state.selected_usecase
#         if "messages" not in st.session_state:
#             st.session_state.messages = []
#         if "thread_id" not in st.session_state:
#             st.session_state.thread_id = f"{st.session_state.selected_usecase}-{uuid.uuid4()}"
#         if "image_bytes" not in st.session_state:
#             st.session_state.image_bytes = None

#     def rotate_thread_and_clear(self, new_usecase: str):
#         # Optional now that we use per-usecase threads, but still good hygiene
#         st.session_state.messages = []
#         st.session_state.thread_id = f"{new_usecase}-{uuid.uuid4()}"
#         st.session_state.image_bytes = None

#     def load_streamlit_ui(self):
#         self.ensure_session_defaults()

#         st.set_page_config(
#             page_title=self.config.get_page_title(),
#             layout="wide",
#         )

#         # --- Global CSS for layout + chat bubbles ---
#         st.markdown(
#             """
#             <style>
#               .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
#               /* Chat bubbles */
#               .chat-user {
#                   background-color: #2a7b37; /* green-ish for dark theme, tweak as you like */
#                   color: #fff;
#                   padding: 10px 14px;
#                   border-radius: 12px;
#                   margin-bottom: 8px;
#                   max-width: 80%;
#                   margin-left: auto;
#                   box-shadow: 0 1px 3px rgba(0,0,0,0.35);
#                   white-space: pre-wrap;
#                   word-wrap: break-word;
#               }
#               .chat-ai {
#                   background-color: #1F2430;
#                   color: #fff;
#                   border: 1px solid #2c3242;
#                   padding: 10px 14px;
#                   border-radius: 12px;
#                   margin-bottom: 8px;
#                   max-width: 80%;
#                   margin-right: auto;
#                   box-shadow: 0 1px 3px rgba(0,0,0,0.35);
#                   white-space: pre-wrap;
#                   word-wrap: break-word;
#               }
#               /* Sidebar cards */
#               section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
#                   padding: 12px;
#                   border-radius: 12px;
#                   background-color: #141821;
#                   margin-bottom: 12px;
#                   border: 1px solid #222838;
#               }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )

#         # --- Product-like Header ---
#         st.markdown(
#             f"""
#             <h1 style='text-align:center; margin-bottom:0.2em;'>🤖 LangGraph Agentic AI</h1>
#             <p style='text-align:center; color:#aab; font-size:16px;'>
#               Stateful · Multi‑Agent · Vision · RAG Chatbot
#             </p>
#             <hr/>
#             """,
#             unsafe_allow_html=True
#         )

#         with st.sidebar:
#             st.markdown("### ⚙️ Configuration")

#             # LLM block
#             with st.container():
#                 st.markdown("**🧠 Language Model**")
#                 llm_options = self.config.get_llm_options()
#                 self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

#                 if self.user_controls["selected_llm"] == "Groq":
#                     model_options = self.config.get_groq_model_options()

#                     # If the user has uploaded an image, filter to vision models for less confusion
#                     if st.session_state.get("image_bytes"):
#                         model_options = [m for m in model_options
#                                          if any(k in m.lower() for k in ("llama-4-scout", "llama-4-maverick"))]

#                     self.user_controls["selected_groq_model"] = st.selectbox(
#                         "Select Groq Model",
#                         model_options,
#                         help="Use Llama‑4 Scout/Maverick for images"
#                     )

#                     self.user_controls["GROQ_API_KEY"] = st.text_input(
#                         "Groq API Key",
#                         type="password",
#                         help="Paste your Groq API key here"
#                     )

#                     if not self.user_controls["GROQ_API_KEY"]:
#                         st.warning("Please enter your Groq API key.")

#             # Use case block
#             st.divider()
#             with st.container():
#                 st.markdown("**🎯 Use Case**")
#                 usecase_options = self.config.get_usecase_options()
#                 selected = st.selectbox(
#                     "Select Use Case",
#                     usecase_options,
#                     index=usecase_options.index(st.session_state.selected_usecase)
#                 )

#                 if selected != st.session_state.selected_usecase:
#                     st.session_state.prev_usecase = st.session_state.selected_usecase
#                     st.session_state.selected_usecase = selected
#                     self.rotate_thread_and_clear(selected)

#                 self.user_controls["selected_usecase"] = st.session_state.selected_usecase

#             # Tiny debug footer (can remove any time)
#             st.caption(
#                 f"Debug → LLM={self.user_controls.get('selected_llm')} · "
#                 f"Model={self.user_controls.get('selected_groq_model')} · "
#                 f"UseCase={self.user_controls.get('selected_usecase')}"
#             )

#         # Use-case badge under header (centered)
#         uc = st.session_state.selected_usecase
#         usecase_icons = {
#             "Basic Chatbot": "💬",
#             "Advanced Chatbot": "🧠",
#             "Multi-Agent Chatbot": "🤖",
#             "Document QA": "📄"
#         }
#         st.markdown(
#             f"""
#             <div style='text-align:center; font-size:20px; margin-bottom:12px; color:#ccd;'>
#               {usecase_icons.get(uc, "✨")} <b>{uc}</b>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#         return self.user_controls
import streamlit as st
from src.langgraphagenticai.ui.uiconfigfile import Config
import uuid


# ── Inline-style constants ───────────────────────────────────────────────────
_GRAD  = "linear-gradient(135deg,#4F7EF7,#7C4FE0)"
_BG2   = "#12151F"
_BOR   = "#1C2035"
_BOR2  = "#1A1D2A"
_TXT1  = "#C8CCDF"
_TXT2  = "#5A5F7A"
_BLUE  = "#4F7EF7"
_NAVY  = "#17203A"
_NAVB  = "#1E2F55"

# ── Global CSS ───────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #0B0D14 !important; color: #E8EAF0 !important; }
.block-container { padding-top: 1rem !important; max-width: 880px !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0E1019 !important; border-right: 1px solid #1A1D2A !important; }
section[data-testid="stSidebar"] label { color: #5A5F7A !important; font-size: 11px !important; font-weight: 500 !important; }
section[data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div {
    background: #0E1019 !important; border: 1px solid #252A40 !important;
    border-radius: 7px !important; color: #C4C8E0 !important; font-size: 13px !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div {
    background: #0E1019 !important; border: 1px solid #252A40 !important;
    border-radius: 7px !important; color: #C4C8E0 !important; font-size: 13px !important;
}
section[data-testid="stSidebar"] .stAlert {
    background: #12151F !important; border: 1px solid #2A1A0A !important;
    border-radius: 7px !important; font-size: 11px !important;
}

/* Chat bubbles */
.chat-wrap { display:flex; flex-direction:column; gap:14px; padding:8px 0 24px; }
.chat-row  { display:flex; gap:10px; max-width:84%; }
.chat-row.user-row { margin-left:auto; flex-direction:row-reverse; }
.chat-avatar {
    width:28px; height:28px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:600; flex-shrink:0; margin-top:2px;
}
.ai-av  { background:linear-gradient(135deg,#4F7EF7,#7C4FE0); color:#fff; }
.usr-av { background:#1C2035; color:#6B7299; }
.bubble { padding:11px 15px; font-size:13px; line-height:1.65; word-wrap:break-word; white-space:pre-wrap; }
.ai-bubble   { background:#12151F; border:1px solid #1C2035; color:#CDD0E3; border-radius:4px 12px 12px 12px; }
.user-bubble { background:#17203A; border:1px solid #1E2F55; color:#B0BFEF; border-radius:12px 4px 12px 12px; }

/* Chat input */
.stChatInput > div { background:#12151F !important; border:1px solid #1C2035 !important; border-radius:12px !important; }
.stChatInput > div:focus-within { border-color:#2D3D70 !important; box-shadow:0 0 0 2px rgba(79,126,247,0.12) !important; }
.stChatInput button { background:linear-gradient(135deg,#4F7EF7,#7C4FE0) !important; border-radius:8px !important; border:none !important; }

/* Misc */
hr { border-color:#1A1D2A !important; }
.stSpinner > div { color:#4F7EF7 !important; }

/* Live dot */
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
.live-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#2ECC71; animation:pulse-dot 2s infinite; vertical-align:middle; margin-right:4px;
}
</style>
"""

USECASE_META = {
    "Basic Chatbot":       {"icon": "💬", "desc": "Single-turn · no memory"},
    "Advanced Chatbot":    {"icon": "🧠", "desc": "Stateful · full history"},
    "Multi-Agent Chatbot": {"icon": "🤖", "desc": "Agent network · tool use"},
    "Document QA":         {"icon": "📄", "desc": "RAG · document grounding"},
}


def _header_html(uc: str, meta: dict) -> str:
    """Fully self-contained header with 100% inline styles — no class dependencies."""
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:10px 0 14px;border-bottom:1px solid {_BOR2};margin-bottom:14px;">'

        # Left: logo + title
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="width:38px;height:38px;border-radius:10px;background:{_GRAD};'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:15px;font-weight:700;color:#fff;flex-shrink:0;">LG</div>'
        f'<div>'
        f'<div style="font-size:17px;font-weight:600;color:{_TXT1};line-height:1.25;">'
        f'LangGraph Agentic AI</div>'
        f'<div style="font-size:11px;color:{_TXT2};margin-top:2px;">'
        f'Stateful &middot; Multi-Agent &middot; Vision &middot; RAG</div>'
        f'</div>'
        f'</div>'

        # Right: badge
        f'<div style="font-size:12px;font-weight:500;color:{_BLUE};'
        f'background:{_NAVY};border:1px solid {_NAVB};'
        f'padding:5px 14px;border-radius:20px;white-space:nowrap;">'
        f'{meta["icon"]}&nbsp;{uc}</div>'

        f'</div>'
    )


def _strip_html(uc: str, meta: dict) -> str:
    """Use-case status strip with 100% inline styles."""
    return (
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'padding:10px 14px;background:{_BG2};border:1px solid {_BOR};'
        f'border-radius:10px;margin-bottom:16px;font-size:13px;color:{_TXT2};">'
        f'<span class="live-dot"></span>'
        f'<span style="font-size:15px;">{meta["icon"]}</span>'
        f'<span style="font-weight:600;color:#A0AACC;">{uc}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:#3E4460;">{meta["desc"]}</span>'
        f'</div>'
    )


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def ensure_session_defaults(self):
        if "selected_usecase" not in st.session_state:
            st.session_state.selected_usecase = "Advanced Chatbot"
        if "prev_usecase" not in st.session_state:
            st.session_state.prev_usecase = st.session_state.selected_usecase
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = f"{st.session_state.selected_usecase}-{uuid.uuid4()}"
        if "image_bytes" not in st.session_state:
            st.session_state.image_bytes = None

    def rotate_thread_and_clear(self, new_usecase: str):
        st.session_state.messages = []
        st.session_state.thread_id = f"{new_usecase}-{uuid.uuid4()}"
        st.session_state.image_bytes = None

    def load_streamlit_ui(self):
        self.ensure_session_defaults()

        st.set_page_config(
            page_title=self.config.get_page_title(),
            page_icon="🤖",
            layout="wide",
        )

        # 1. Inject global CSS
        st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

        # 2. Header
        uc   = st.session_state.selected_usecase
        meta = USECASE_META.get(uc, {"icon": "✨", "desc": ""})
        st.markdown(_header_html(uc, meta), unsafe_allow_html=True)

        # 3. Sidebar
        with st.sidebar:
            st.markdown(
                "<p style='font-size:10px;font-weight:600;letter-spacing:1.2px;"
                "color:#3E4460;text-transform:uppercase;margin:4px 0 12px;'>"
                "⚙&nbsp; Configuration</p>",
                unsafe_allow_html=True,
            )

            # ── LLM Provider ────────────────────────────────────────────────
            st.markdown(
                f"<p style='font-size:12px;font-weight:600;color:{_TXT1};margin:0 0 6px;'>"
                "🧠 Language Model</p>",
                unsafe_allow_html=True,
            )
            llm_options = self.config.get_llm_options()
            self.user_controls["selected_llm"] = st.selectbox("Provider", llm_options)

            if self.user_controls["selected_llm"] == "Groq":

                # ── Full model list ──────────────────────────────────────────
                # CHANGE 1: updated model list — 6 current free Groq models
                model_options = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "qwen/qwen3-32b",
                    "openai/gpt-oss-120b",
                    "meta-llama/llama-4-scout-17b-16e-instruct",  # vision
                    "llama-3.2-11b-vision-preview",               # vision
                ]

                # ── CHANGE 2: filter to vision-only when image is attached ──
                # Use short substrings — full path strings won't match after lower()
                if st.session_state.get("image_bytes"):
                    model_options = [
                        m for m in model_options
                        if any(k in m.lower() for k in ("scout", "vision-preview"))
                    ]

                self.user_controls["selected_groq_model"] = st.selectbox(
                    "Model",
                    model_options,
                    help="Scout / Vision-Preview support image inputs",
                )

                chosen = self.user_controls.get("selected_groq_model", "")

                # ── CHANGE 3: vision badge — maverick removed, vision-preview added ──
                if any(k in chosen.lower() for k in ("scout", "vision-preview")):
                    st.markdown(
                        f"<div style='font-size:10px;background:#1A2540;border:1px solid #223060;"
                        f"border-radius:5px;padding:3px 8px;color:{_BLUE};font-weight:500;"
                        f"display:inline-block;margin-bottom:6px;'>Vision enabled ✦</div>",
                        unsafe_allow_html=True,
                    )

                # ── Model description pills ──────────────────────────────────
                model_notes = {
                    "llama-3.3-70b-versatile":                    ("⭐ Best free · 128k ctx",  "#17203A", "#4F7EF7", "#1E2F55"),
                    "llama-3.1-8b-instant":                       ("⚡ Fastest · 128k ctx",    "#17201A", "#2ECC71", "#1E3A20"),
                    "qwen/qwen3-32b":                              ("🧠 Reasoning · 131k ctx",  "#1A1730", "#A77FE8", "#2A1E50"),
                    "openai/gpt-oss-120b":                        ("🔥 Flagship · reasoning",  "#1A1520", "#E8A87C", "#3A2A10"),
                    "meta-llama/llama-4-scout-17b-16e-instruct":  ("🖼 Vision · 128k ctx",     "#1A0F38", "#A77FE8", "#3A1E60"),
                    "llama-3.2-11b-vision-preview":               ("🖼 Vision lite · preview", "#1A0F38", "#A77FE8", "#3A1E60"),
                }
                if chosen in model_notes:
                    label, bg, fg, bor = model_notes[chosen]
                    st.markdown(
                        f"<div style='font-size:10px;background:{bg};border:1px solid {bor};"
                        f"border-radius:5px;padding:3px 8px;color:{fg};font-weight:500;"
                        f"display:inline-block;margin-bottom:6px;'>{label}</div>",
                        unsafe_allow_html=True,
                    )

                self.user_controls["GROQ_API_KEY"] = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="gsk_••••••••••",
                    help="Your Groq API key — never stored",
                )
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Enter your Groq API key to enable the chatbot.")

            st.divider()

            # ── Use Case ─────────────────────────────────────────────────────
            st.markdown(
                f"<p style='font-size:12px;font-weight:600;color:{_TXT1};margin:0 0 6px;'>"
                "🎯 Use Case</p>",
                unsafe_allow_html=True,
            )
            usecase_options = self.config.get_usecase_options()
            selected = st.selectbox(
                "Mode",
                usecase_options,
                index=usecase_options.index(st.session_state.selected_usecase),
            )
            if selected != st.session_state.selected_usecase:
                st.session_state.prev_usecase = st.session_state.selected_usecase
                st.session_state.selected_usecase = selected
                self.rotate_thread_and_clear(selected)

            self.user_controls["selected_usecase"] = st.session_state.selected_usecase

            st.divider()
            tid = st.session_state.thread_id
            st.caption(f"Thread · {tid[:28]}…")

        # 4. Use-case strip (re-read after sidebar interaction)
        uc   = st.session_state.selected_usecase
        meta = USECASE_META.get(uc, {"icon": "✨", "desc": ""})
        st.markdown(_strip_html(uc, meta), unsafe_allow_html=True)

        return self.user_controls