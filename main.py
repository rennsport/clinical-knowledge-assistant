import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

import gradio as gr

from pathlib import Path
import requests
from urllib.parse import urlparse

# Load variables from .env in project root if present
loaded = load_dotenv()
load_dotenv("./prompts.env")

# Validate key presence
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Add it to .env (or prompts.env) or export it in your shell."
    )


model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")

model = ChatOpenAI(model=model_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

vector_store = InMemoryVectorStore(embeddings)

# Local cache directory and urls list file
documents_dir = (Path.cwd() / "documents").resolve()
documents_dir.mkdir(parents=True, exist_ok=True)
urls_file = documents_dir / "urls.txt"

# Read URLs from file (one per line, '#' for comments); fallback to default if empty
urls = []
for line in urls_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    urls.append(line)

# Download/cache each URL into documents/
local_paths = []
for url in urls:
    name = Path(urlparse(url).path).name or "document.pdf"
    local_path = documents_dir / name
    if not local_path.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
    local_paths.append(local_path)

loaders = [PyPDFLoader(str(p)) for p in local_paths]

# Load all documents from the loaders list
all_docs = []
for ldr in loaders:
    try:
        loaded = ldr.load()
        all_docs.extend(loaded)
    except Exception as e:
        print(f"Failed to load {ldr.file_path}: {e}")

docs = all_docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

def rag_agent(message, history):
    tools = [retrieve_context]
    prompt = os.getenv("SYSTEM_PROMPT")
    agent = create_agent(model, tools, system_prompt=prompt)

    final_text = None
    for event in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values",
    ):
        # capture only the latest assistant content to return to gradio
        final_text = event["messages"][-1].content

    return final_text or "No response generated."



app = gr.ChatInterface(
    fn=rag_agent,
    title="RAG Chatbot",
    description="Ask me anything about the loaded documents!",
)

app.launch()