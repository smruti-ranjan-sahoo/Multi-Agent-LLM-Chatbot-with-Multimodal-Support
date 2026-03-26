# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

# def create_vector_store(chunks):
#     """
#     Create a FAISS vector store from document chunks.
#     """

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vector_store = FAISS.from_documents(
#         documents=chunks,
#         embedding=embeddings
#     )

#     return vector_store
from langchain_community.embeddings import JinaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()


def create_vector_store(chunks, namespace: str):

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
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
        jina_api_key=os.getenv("JINA_API_KEY")
    )

    # vector_store = PineconeVectorStore.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     index_name=index_name
    # )
    vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace=namespace   # ✅ isolates data
)

    return vector_store
