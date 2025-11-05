import requests
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_urls(urls_file: Path) -> list[str]:
    """Read newline-delimited URLs from ``urls_file``."""
    if not urls_file.exists():
        return []
    urls: list[str] = []
    for line in urls_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def download_documents(urls: list[str], destination_dir: Path) -> list[Path]:
    """Download documents to ``destination_dir`` if not already present.

    Skips files that already exist so repeated runs are fast and safe. 
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for url in urls:
        name = Path(urlparse(url).path).name or "document.pdf"
        local_path = destination_dir / name
        if not local_path.exists():
            try:
                print(f"[Download] Downloading {url}...")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
        local_paths.append(local_path)
    return local_paths


def load_pdfs(paths: list[Path]) -> list:
    """Load PDFs into LangChain ``Document`` objects.

    Uses ``PyPDFLoader`` for parsing. Logs progress for long
    ingestions to ensure the user knows the step of the process they're on
    """
    print(f"[Load] Loading {len(paths)} PDFs...")
    loaders = [PyPDFLoader(str(p)) for p in paths]
    all_docs = []
    for loader in loaders:
        try:
            print(f"[Load] Loading {loader.file_path}...")
            loaded_docs = loader.load()
            all_docs.extend(loaded_docs)
        except Exception as e:
            file_label = getattr(loader, "file_path", str(loader))
            print(f"Failed to load {file_label}: {e}")
    return all_docs


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into overlapping chunks suitable for retrieval.

    ``RecursiveCharacterTextSplitter`` preserves semantic continuity via
    overlap. Chunk sizes are configurable via env to tune recall vs. latency.
    """
    print(f"[Split] Splitting {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_vector_store(splits: list, embedding_model: str = "text-embedding-3-large") -> InMemoryVectorStore:
    """Create an in-memory vector store from chunks using OpenAI embeddings.

    ``InMemoryVectorStore`` keeps the POC simple and dependency-free.
    Swapping to a persistent store (FAISS, Chroma, PGVector) only requires changing this
    function. Embedding model is injected to allow testing multiple.
    """
    print(f"[Build] Building vector store with {len(splits)} splits...")
    store = InMemoryVectorStore(OpenAIEmbeddings(model=embedding_model))
    if splits:
        store.add_documents(documents=splits)
    return store
