from langchain_core.messages import SystemMessage
from src.langgraphagenticai.state.state import State


class RAGAgent:
    """
    Retrieval-Augmented Generation agent.
    Uses a retriever to fetch relevant documents
    and answers using the LLM.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = SystemMessage(
            content=(
                "You are a document question-answering assistant. "
                "Use the provided context to answer the user's question. "
                "If the answer is not in the context, say you don't know."
            )
        )

    # def process(self, state: State) -> dict:
    #     query = state["messages"][-1].content

    #     docs = self.retriever.invoke(query)

    #     context = "\n\n".join(doc.page_content for doc in docs)

    #     messages = [
    #         self.system_prompt,
    #         SystemMessage(content=f"Context:\n{context}"),
    #         *state["messages"],
    #     ]

    #     response = self.llm.invoke(messages)

    #     # return {"messages": [response]}
    #     return {"messages": state["messages"] + [response]}
    def process(self, state: State) -> dict:
        query = state["messages"][-1].content

        docs = self.retriever.invoke(query)

        # 🔍 DEBUG: see what is actually retrieved
        print("\n🔍 Retrieved Docs:")
        for d in docs:
            print(d.metadata, d.page_content[:100])

        context = "\n\n".join(doc.page_content for doc in docs)

        messages = [
            self.system_prompt,
            SystemMessage(content=f"Context:\n{context}"),
            *state["messages"],
        ]

        response = self.llm.invoke(messages)

        return {"messages": state["messages"] + [response]}