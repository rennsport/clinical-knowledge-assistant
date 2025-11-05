import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from agent_utils import make_rag_agent

from document_utils import (
    read_urls,
    download_documents,
    load_pdfs,
    split_documents,
    build_vector_store,
)
from gradio_app import create_app


def load_environment() -> None:
    """Load environment variables from .env and prompts.env.

    Supports a secondary ``prompts.env`` to allow prompt iteration
    without touching secrets. Fails fast if ``OPENAI_API_KEY`` is missing.
    """
    load_dotenv()
    load_dotenv("./prompts.env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            (
                "OPENAI_API_KEY not set. Add it to .env (or prompts.env) "
                "or export it in your shell."
            )
        )


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
DOCUMENTS_DIR = (Path.cwd() / "documents").resolve()
URLS_FILE = DOCUMENTS_DIR / "urls.txt"


def initialize():
    """Build the retrieval pipeline and return a bound chat function.

    Constructs dependencies (docs → splits → vector
    store, then model) and returns an agent created by ``make_rag_agent``.
    """
    load_environment()

    urls = read_urls(URLS_FILE)
    local_paths = download_documents(urls, DOCUMENTS_DIR) if urls else []
    docs = load_pdfs(local_paths) if local_paths else []
    splits = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP) if docs else []
    vector_store = build_vector_store(splits, EMBEDDING_MODEL)

    model = ChatOpenAI(model=MODEL_NAME)
    return make_rag_agent(model, vector_store, MAX_HISTORY_TURNS)


if __name__ == "__main__":
    rag_fn = initialize()
    app = create_app(rag_fn)
    app.launch()
