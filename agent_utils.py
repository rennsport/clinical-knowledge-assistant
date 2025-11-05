import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI


def make_retrieve_context_tool(current_vector_store: InMemoryVectorStore):
    """Bind a retrieval tool to a specific vector store.

    Returns serialized context and artifacts for downstream usage.
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


def make_rag_agent(
    current_model: ChatOpenAI,
    current_vector_store: InMemoryVectorStore,
    max_history_turns: int = 5,
):
    """Create a streaming RAG agent bound to the provided model/store.

    Construct once and return a lightweight callable agent compatible with
    Gradio's interface. By constructing once and returning a function, I avoid
    creating a new agent for each request.
    """
    tools = [make_retrieve_context_tool(current_vector_store)]
    prompt = os.getenv("SYSTEM_PROMPT")
    agent = create_agent(current_model, tools, system_prompt=prompt)

    def rag_agent(message, history):
        """Chat handler used by the UI layer; streams final text only.

        Uses up to ``max_history_turns`` previous messages for context.
        """
        # Prepare messages with truncated history (type="messages")
        messages = []
        if history:
            # Gradio ChatInterface type="messages": list of {role, content}
            recent = history[-max_history_turns:]
            for item in recent:
                if (
                    isinstance(item, dict)
                    and "role" in item
                    and "content" in item
                ):
                    messages.append(
                        {
                            "role": str(item["role"]),
                            "content": str(item["content"]),
                        }
                    )

        # Append the current turn
        if (
            isinstance(message, dict)
            and "role" in message
            and "content" in message
        ):
            messages.append(
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
            )
        else:
            messages.append({"role": "user", "content": str(message)})

        final_text = None
        for event in agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            final_text = event["messages"][-1].content
        return final_text or "No response generated."

    return rag_agent
