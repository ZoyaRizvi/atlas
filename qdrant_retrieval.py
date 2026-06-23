from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse


class RetrievalError(Exception):
    pass


def normalize_qdrant_url(url: str) -> str:
    if not url:
        return "http://localhost:6333"

    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse(f"http://{url}")

    if not parsed.hostname:
        raise RetrievalError(f"Invalid Qdrant URL: {url}")

    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{parsed.hostname}:{parsed.port}"

    return urlunparse((parsed.scheme, netloc, parsed.path or "", "", "", ""))


def _raise_connection_error(exc: Exception, qdrant_url: str) -> None:
    message = str(exc)
    if (
        isinstance(exc, (ConnectionError, TimeoutError))
        or "Connection refused" in message
        or "Failed to establish a new connection" in message
        or "Name or service not known" in message
        or "getaddrinfo" in message
    ):
        raise RetrievalError(
            f"Unable to connect to Qdrant at {qdrant_url}. "
            "Make sure Qdrant is running and reachable, for example: "
            "docker run -p 6333:6333 qdrant/qdrant"
        ) from exc
    raise RetrievalError(str(exc)) from exc


def load_documents_from_folder(folder_path: str) -> list:
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
    except ImportError as exc:
        raise RetrievalError(
            "Missing LangChain document loader dependencies. "
            "Install with: pip install langchain-community PyPDF2"
        ) from exc

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise RetrievalError(f"Folder not found: {folder_path}")

    documents = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".txt", ".md", ".csv", ".json"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())

    if not documents:
        raise RetrievalError(
            "No supported documents found in the selected folder. "
            "Supported extensions: .pdf, .txt, .md, .csv, .json"
        )

    return documents


def get_embedding_model():
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        OpenAIEmbeddings = None

    try:
        if OpenAIEmbeddings is not None and os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbeddings()
    except Exception:
        pass

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError as exc:
            raise RetrievalError(
                "No embedding backend is available. "
                "Install sentence-transformers for HuggingFaceEmbeddings or set OPENAI_API_KEY for OpenAIEmbeddings."
            ) from exc


def get_llm():
    if os.environ.get("XAI_API_KEY"):
        try:
            from langchain_openai import OpenAI
            api_key = os.environ.get("XAI_API_KEY")
            return OpenAI(
                model_name="grok-2",
                temperature=0,
                openai_api_key=api_key,
                openai_api_base="https://api.x.ai/v1",
            )
        except ImportError as exc:
            raise RetrievalError(
                "OpenAI package is not installed. "
                "Install with: pip install langchain-openai"
            ) from exc

    raise RetrievalError(
        "XAI_API_KEY environment variable is not set. Set it with: export XAI_API_KEY='xai-...'"
    )


def build_qdrant_collection(
    folder_path: str,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
) -> None:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Qdrant
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RetrievalError(
            "Missing Qdrant or LangChain dependencies. "
            "Install with: pip install langchain qdrant-client langchain-community"
        ) from exc

    qdrant_url = normalize_qdrant_url(qdrant_url)
    documents = load_documents_from_folder(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    embeddings = get_embedding_model()
    try:
        client = QdrantClient(url=qdrant_url, api_key=None)
        collections = [collection.name for collection in client.get_collections().collections]
        if collection_name in collections:
            client.delete_collection(collection_name=collection_name)
        Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name,
            prefer_grpc=False,
        )
    except Exception as exc:
        _raise_connection_error(exc, qdrant_url)


def build_retrieval_chain(
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
):
    try:
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import Qdrant
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RetrievalError(
            "Missing Qdrant or LangChain dependencies. "
            "Install with: pip install langchain qdrant-client langchain-community"
        ) from exc

    qdrant_url = normalize_qdrant_url(qdrant_url)

    embeddings = get_embedding_model()
    llm = get_llm()
    try:
        client = QdrantClient(url=qdrant_url, api_key=None)
        store = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
    except Exception as exc:
        _raise_connection_error(exc, qdrant_url)

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )


def answer_query_with_sources(chain, query: str) -> tuple[str, list]:
    result = chain({
        "query": query,
    })
    answer = result.get("result") or result.get("answer") or ""
    sources = result.get("source_documents") or []
    return answer, sources


def format_sources(documents: list) -> str:
    if not documents:
        return ""

    lines: list[str] = []
    seen_sources: set[str] = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        if source in seen_sources:
            continue
        seen_sources.add(source)

        label = source
        page_info = doc.metadata.get("page")
        if page_info is not None:
            label = f"{Path(source).name}:page {page_info}"
        elif Path(source).name:
            label = Path(source).name

        snippet = " ".join(doc.page_content.strip().split())
        if len(snippet) > 150:
            snippet = snippet[:150].rstrip() + "..."

        lines.append(f"- {label} — {snippet}")

    return "\n".join(lines)
