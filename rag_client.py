# Rag_client.py
#found reference for this import via google search
from __future__ import annotations

import os
from typing import Any, Tuple
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    candidate_dirs = [p for p in current_dir.iterdir() if p.is_dir() and ("chroma" in p.name.lower())]

    for d in candidate_dirs:
        try:
            chromadb, Settings = _get_chromadb()
            client = chromadb.PersistentClient(
                path=str(d),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()

            for col in collections:
                col_name = getattr(col, "name", str(col))
                collection = client.get_collection(col_name)

                key = f"{d.name}::{col_name}"
                try:
                    count = collection.count()
                except Exception:
                    count = -1

                backends[key] = {
                    "directory": str(d),
                    "collection_name": col_name,
                    "display_name": f"{d.name} / {col_name}" + (f" ({count} docs)" if count >= 0 else ""),
                    "count": str(count),
                }
        except Exception as e:
            err = str(e).replace("\n", " ")
            if len(err) > 80:
                err = err[:77] + "..."
            key = f"{d.name}::<error>"
            backends[key] = {
                "directory": str(d),
                "collection_name": "",
                "display_name": f"{d.name} (unavailable: {err})",
                "count": "-1",
            }

    return backends


def _get_chromadb() -> Tuple[Any, Any]:
    """Lazy import ChromaDB and its Settings. Returns (chromadb, Settings).

    Raises ImportError with a helpful message if ChromaDB is not installed
    or incompatible with the Python runtime.
    """
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb, Settings
    except Exception as e:
        raise ImportError(
            "ChromaDB is required for this operation but cannot be imported. "
            "Install chromadb (pip install chromadb) or use a compatible Python version. "
            f"Underlying error: {e}"
        ) from e


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend.

    Returns the collection object, or `None` if ChromaDB is not available.
    """
    try:
        chromadb, Settings = _get_chromadb()
    except ImportError:
        return None

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(collection_name)


def _embed_query(query: str) -> List[float]:
    """Embed the query using OpenAI embeddings for semantic search."""
    api_key = os.getenv("OPENAI_API_KEY", "voc-2009866425126677464339669333e56cd9164.50366416")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set (required for query embedding).")

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)

    resp = client.embeddings.create(model=embed_model, input=[query])
    return resp.data[0].embedding


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering (and embedded queries)."""

    where = None
    if mission_filter and str(mission_filter).strip().lower() not in {"all", "none", ""}:
        where = {"mission": mission_filter}

    query_embedding = _embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context with deduplication."""
    if not documents:
        return ""

    #preserving order and metadata alignment
    seen = set()
    deduped_docs = []
    deduped_metas = []
    for doc, meta in zip(documents, metadatas):
        if doc not in seen:
            seen.add(doc)
            deduped_docs.append(doc)
            deduped_metas.append(meta)

    context_parts: List[str] = ["### Retrieved NASA Archive Context\n"]

    for idx, (doc, meta) in enumerate(zip(deduped_docs, deduped_metas), start=1):
        meta = meta or {}

        mission = meta.get("mission", "unknown_mission")
        mission_clean = str(mission).replace("_", " ").title()

        source = meta.get("source", "unknown_source")

        category = meta.get("category", "document")
        category_clean = str(category).replace("_", " ").title()

        header = f"Source {idx} | Mission: {mission_clean} | Category: {category_clean} | File: {source}"
        context_parts.append(header)

        doc = doc or ""
        max_chars = 1600
        if len(doc) > max_chars:
            doc = doc[:max_chars].rstrip() + "..."

        context_parts.append(doc)
        context_parts.append("")

    return "\n".join(context_parts).strip()