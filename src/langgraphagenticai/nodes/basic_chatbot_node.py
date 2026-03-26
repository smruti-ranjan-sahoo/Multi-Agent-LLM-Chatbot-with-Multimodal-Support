from src.langgraphagenticai.state.state import State

from langchain_core.messages import SystemMessage

class BasicChatbotNode:
    def __init__(self, model):
        self.llm = model
        self.system_prompt = SystemMessage(
            content=(
                "You are an expert AI assistant. "
                "Answer clearly, concisely, and professionally. "
                "If you do not know something, say so explicitly."
            )
        )

    def process(self, state: State) -> dict:
        messages = state["messages"]

        # Inject system prompt only once
        if not any(msg.type == "system" for msg in messages):
            messages = [self.system_prompt] + messages

        response = self.llm.invoke(messages)
        return {"messages": response}
