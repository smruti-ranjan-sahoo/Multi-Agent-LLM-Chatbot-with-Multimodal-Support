from langchain_community.embeddings import JinaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import streamlit as st


def create_vector_store(chunks, namespace: str):

    # 🔥 Use Streamlit secrets instead of .env
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = "chatbot"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    embeddings = JinaEmbeddings(
        model_name="jina-embeddings-v2-base-en",
        jina_api_key=st.secrets["JINA_API_KEY"]
    )

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace   # ✅ isolate per user/thread
    )

    return vector_store