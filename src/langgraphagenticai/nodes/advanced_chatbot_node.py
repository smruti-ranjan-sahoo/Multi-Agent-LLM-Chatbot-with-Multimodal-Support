from langchain_core.messages import SystemMessage
from src.langgraphagenticai.state.state import State

class AdvancedChatbotNode:
    def __init__(self, model):
        self.llm = model
        self.system_prompt = SystemMessage(
            content=(
                "You are an advanced AI assistant. "
                "Provide structured, detailed answers. "
                "Ask clarifying questions if needed."
            )
        )

    def process(self, state: State) -> dict:
        messages = state["messages"]

        if not any(msg.type == "system" for msg in messages):
            messages = [self.system_prompt] + messages

        response = self.llm.invoke(messages)
        return {"messages": response}
