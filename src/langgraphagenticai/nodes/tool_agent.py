from typing import Dict, Any, List, Callable
from langchain_core.messages import AIMessage
from src.langgraphagenticai.state.state import State


class ToolAgent:
    def __init__(self, tools=None):
        # expect list or dict; adapt to your usage
        self.tools = tools or []

    def process(self, state: State) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        image = state.get("image")
        plan = state.get("plan")

        # Implement your tool dispatch here
        tool_output = None
        if isinstance(self.tools, dict) and "search" in self.tools and plan and "search" in plan.lower():
            tool_output = self.tools["search"]("query from plan")

        # Add tool result as assistant content (optional)
        if tool_output:
            msgs = list(msgs) + [AIMessage(content=f"[TOOL]: {tool_output}")]

        return {
            "messages": msgs,
            "plan": plan,
            "tool_result": tool_output,
            "image": image,  # PRESERVE IMAGE
        }