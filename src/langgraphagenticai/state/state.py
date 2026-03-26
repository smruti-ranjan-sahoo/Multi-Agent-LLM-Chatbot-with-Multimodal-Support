from typing import Annotated, List, TypedDict, Optional
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Shared state across LangGraph nodes.

    messages     : Conversation history (all use cases)
    plan         : PlannerAgent output (multi-agent only)
    tool_result  : ToolAgent output (multi-agent only)
    image        : Optional uploaded image bytes (multi-agent only)
    
    """
    messages: Annotated[List, add_messages]
    plan: Optional[str]
    tool_result: Optional[str]
    image: Optional[bytes]