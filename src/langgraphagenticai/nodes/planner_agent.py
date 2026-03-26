from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.langgraphagenticai.state.state import State


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def process(self, state: State) -> Dict[str, Any]:
        messages = state.get("messages", [])
        image = state.get("image")

        sys = SystemMessage(content="You are a planner. Decide if tools are needed.")
        prompt = [sys] + messages
        plan_ai = self.llm.invoke(prompt)
        plan_text = getattr(plan_ai, "content", "No plan.")

        # Append planner's system-style feedback
        messages_out = list(messages) + [AIMessage(content=f"[PLAN]: {plan_text}")]

        return {
            "messages": messages_out,
            "plan": plan_text,
            "tool_result": state.get("tool_result"),
            "image": image,  #PRESERVE IMAGE
        }