from langgraph.graph import StateGraph, START, END
from src.langgraphagenticai.state.state import State

from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.advanced_chatbot_node import AdvancedChatbotNode
from src.langgraphagenticai.nodes.planner_agent import PlannerAgent
from src.langgraphagenticai.nodes.tool_agent import ToolAgent
from src.langgraphagenticai.nodes.chat_agent import ChatAgent
from src.langgraphagenticai.nodes.rag_agent import RAGAgent


class GraphBuilder:
    def __init__(self, llm, tools=None, retriever=None):
        self.llm = llm
        # Prefer list by default; ToolAgent handles both
        self.tools = tools or []
        self.retriever = retriever

    def basic_chatbot(self):
        graph = StateGraph(State)
        node = BasicChatbotNode(self.llm)
        graph.add_node("basic_chat", node.process)
        graph.add_edge(START, "basic_chat")
        graph.add_edge("basic_chat", END)
        return graph

    def advanced_chatbot(self):
        graph = StateGraph(State)
        node = AdvancedChatbotNode(self.llm)
        graph.add_node("advanced_chat", node.process)
        graph.add_edge(START, "advanced_chat")
        graph.add_edge("advanced_chat", END)
        return graph

    def multi_agent_chatbot(self):
        graph = StateGraph(State)
        planner = PlannerAgent(self.llm)
        tooler = ToolAgent(self.tools)
        chatter = ChatAgent(self.llm)

        graph.add_node("planner", planner.process)
        graph.add_node("tool", tooler.process)
        graph.add_node("chat", chatter.process)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "tool")
        graph.add_edge("tool", "chat")
        graph.add_edge("chat", END)
        return graph

    def document_qa(self):
        if self.retriever is None:
            raise ValueError("Retriever must be provided for Document QA")
        graph = StateGraph(State)
        rag_node = RAGAgent(self.llm, self.retriever)
        graph.add_node("rag", rag_node.process)
        graph.add_edge(START, "rag")
        graph.add_edge("rag", END)
        return graph

    def setup_graph(self, usecase: str):
        if usecase == "Basic Chatbot":
            graph = self.basic_chatbot()
        elif usecase == "Advanced Chatbot":
            graph = self.advanced_chatbot()
        elif usecase == "Multi-Agent Chatbot":
            graph = self.multi_agent_chatbot()
        elif usecase == "Document QA":
            graph = self.document_qa()
        else:
            raise ValueError(f"Unknown usecase: {usecase}")
        return graph.compile()