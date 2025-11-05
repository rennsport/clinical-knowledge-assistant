import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI


def make_retrieve_context_tool(current_vector_store: InMemoryVectorStore):
    """Function that binds a retrieval tool to a specific vector store.

    The tool returns both serialized context and artifacts for downstream usage.
    """
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        if current_vector_store is None:
            return "Vector store is not initialized.", []
        retrieved_docs = current_vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
            ) for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


def make_rag_agent(current_model: ChatOpenAI,
    current_vector_store: InMemoryVectorStore,):
    """Create a streaming RAG agent bound to the provided model/store.

    Construct once and return a lightweight callable agent compatible with
    Gradio's interface. By constructing once and returning a function, I avoid
    creating a new agent for each request.
    """
    tools = [make_retrieve_context_tool(current_vector_store)]
    prompt = os.getenv("SYSTEM_PROMPT")
    agent = create_agent(current_model, tools, system_prompt=prompt)

    def rag_agent(message, history):
        """Chat handler used by the UI layer; streams final text only."""
        final_text = None
        for event in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
        ):
            final_text = event["messages"][-1].content
        return final_text or "No response generated."

    return rag_agent
