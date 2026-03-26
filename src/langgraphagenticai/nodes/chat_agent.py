from typing import Dict, Any, List, Optional
import base64
import imghdr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from src.langgraphagenticai.state.state import State


def guess_mime_from_bytes(data: bytes) -> str:
    kind = imghdr.what(None, data)
    if kind == "png":
        return "image/png"
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    if kind == "gif":
        return "image/gif"
    if kind == "webp":
        return "image/webp"
    return "image/png"


def to_data_url(image_bytes: bytes) -> str:
    mime = guess_mime_from_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def is_groq_vision_model(llm) -> bool:
    name = (getattr(llm, "model", "") or getattr(llm, "model_name", "") or "").lower()
    return any(k in name for k in ("llama-4-scout", "llama-4-maverick", "vision"))


class ChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.base_prompt = (
            "You are a chat agent in a multi-agent system.\n"
            "Rules:\n"
            "- Do NOT invent tools, models, or analysis pipelines.\n"
            "- If an image is provided but no tool output exists, acknowledge the image and ask the user what they want to do.\n"
            "- Be concise and factual.\n"
            "- Use tool results ONLY if they are explicitly provided.\n"
        )

    def build_system_context(self, state: State) -> List[BaseMessage]:
        msgs: List[BaseMessage] = [SystemMessage(content=self.base_prompt)]
        if state.get("plan"):
            msgs.append(SystemMessage(content=f"Planner decision: {state['plan']}"))
        if state.get("tool_result"):
            msgs.append(SystemMessage(content=f"Tool result: {state['tool_result']}"))
        return msgs

    def latest_user_message(self, messages: List[BaseMessage]) -> Optional[HumanMessage]:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m
        return None

    def user_wants_image_analysis(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in (
            "explain", "describe", "analyze", "analyse", "what is", "what’s", "what's",
            "summarize", "read", "ocr", "extract","answer", "question", "text in image", "details"
        ))

    def process(self, state: State) -> Dict[str, Any]:
        history: List[BaseMessage] = state.get("messages", [])
        image_bytes: Optional[bytes] = state.get("image")
        plan = state.get("plan")
        tool_result = state.get("tool_result")

        # If image present but no tool output yet → check user's intent
        if image_bytes is not None and not tool_result:
            latest_user = self.latest_user_message(history)
            wants = self.user_wants_image_analysis(latest_user.content if latest_user else "")
            if not wants:
                response = AIMessage(
                    content=(
                        "I’ve received the image.\n\n"
                        "What would you like me to do with it?\n"
                        "- Describe it\n"
                        "- Extract text\n"
                        "- Answer questions about it\n"
                    )
                )
                return {
                    "messages": history + [response],
                    "image": image_bytes,
                    "plan": plan,
                    "tool_result": tool_result,
                }

        # Build LLM messages
        llm_msgs: List[BaseMessage] = self.build_system_context(state)

        if image_bytes is not None:
            if is_groq_vision_model(self.llm):
                latest_user = self.latest_user_message(history)
                user_text = latest_user.content if latest_user else "Please analyze the image."
                data_url = to_data_url(image_bytes)

                llm_msgs.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                )
            else:
                # Graceful feedback if a text-only model is selected
                response = AIMessage(
                    content=(
                        "I have your image, but the selected model doesn't support vision. "
                        "Please switch to a Groq vision model such as "
                        "`meta-llama/llama-4-scout-17b-16e-instruct` or "
                        "`meta-llama/llama-4-maverick-17b-128e-instruct`."
                    )
                )
                return {
                    "messages": history + [response],
                    "image": image_bytes,
                    "plan": plan,
                    "tool_result": tool_result,
                }
        else:
            llm_msgs.extend(history)

        # Debug
        print("[ChatAgent] model:", getattr(self.llm, "model", None),
              "img?", image_bytes is not None)

        # Call LLM
        ai = self.llm.invoke(llm_msgs)
        ai_msg = ai if isinstance(ai, AIMessage) else AIMessage(content=getattr(ai, "content", str(ai)))

        return {
            "messages": history + [ai_msg],  # append
            "image": image_bytes,             # preserve image
            "plan": plan,
            "tool_result": tool_result,
        }