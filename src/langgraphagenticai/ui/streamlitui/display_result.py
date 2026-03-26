import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class DisplayResultStreamlit:
    """
    Renders and updates memory for stateful modes:
    - Advanced Chatbot  : full history sent to graph
    - Multi-Agent / Doc QA: latest user msg only (+ image / query)
    """

    def __init__(self, usecase, graph, messages, image=None, thread_id: str | None = None):
        self.usecase   = usecase
        self.graph     = graph
        self.messages  = messages
        self.image     = image
        self.thread_id = thread_id

    # ── helpers ──────────────────────────────────────────────────────────
    def latest_user(self):
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                return m
        return None

    def build_state_for_usecase(self):
        latest = self.latest_user()
        if self.usecase == "Advanced Chatbot":
            state = {"messages": list(self.messages)}
        elif self.usecase in ("Multi-Agent Chatbot", "Document QA"):
            state = {"messages": [latest] if latest else []}
            if self.usecase == "Document QA" and latest:
                state["query"] = latest.content
        else:
            state = {"messages": [latest] if latest else []}

        if self.image is not None:
            state["image"] = self.image
        return state

    def invoke(self, state):
        cfg = {"configurable": {"thread_id": self.thread_id}} if self.thread_id else {}
        return self.graph.invoke(state, config=cfg) if cfg else self.graph.invoke(state)

    # ── render a single message bubble ───────────────────────────────────
    @staticmethod
    def _render_bubble(msg: BaseMessage):
        if isinstance(msg, HumanMessage):
            st.markdown(
                f"""
                <div class="chat-row user-row">
                  <div class="chat-avatar usr-av">U</div>
                  <div class="bubble user-bubble">{msg.content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif isinstance(msg, AIMessage):
            st.markdown(
                f"""
                <div class="chat-row">
                  <div class="chat-avatar ai-av">AI</div>
                  <div class="bubble ai-bubble">{msg.content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── main entry point ─────────────────────────────────────────────────
    def display_result_on_ui(self):
        state = self.build_state_for_usecase()

        with st.spinner("Thinking…"):
            result = self.invoke(state)

        # extract AI reply
        ai_msg  = None
        msgs    = result.get("messages", [])
        if msgs and isinstance(msgs[-1], AIMessage):
            ai_msg = msgs[-1]
        else:
            txt = result.get("reply") or result.get("answer")
            if txt:
                ai_msg = AIMessage(content=txt)
        if not ai_msg:
            return

        self.messages.append(ai_msg)

        # render full transcript in styled bubbles
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in self.messages:
            self._render_bubble(msg)
        st.markdown("</div>", unsafe_allow_html=True)