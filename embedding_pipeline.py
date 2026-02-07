# embedding.py
# embedding_pipeline.py
# removed some duplicate code and added some error handling for missing dependencies. Also added more comments for clarity.
# CoPilot found issues with my code compared to the scoring matrix.
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

from openai import OpenAI

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").split())


def iter_documents(data_dir: str) -> List[Tuple[str, str, str]]:
    """Returns list of (mission, file_path, text). Expects data/<mission>/**/*.txt or .md"""
    base = Path(data_dir)
    docs: List[Tuple[str, str, str]] = []
    for mission_dir in base.iterdir():
        if not mission_dir.is_dir():
            continue
        mission = mission_dir.name
        for fp in mission_dir.rglob("*"):
            if fp.suffix.lower() not in [".txt", ".md"]:
                continue
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            docs.append((mission, str(fp), txt))
    return docs


def _get_encoder():
    if not _HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _get_chromadb() -> Tuple[Any, Any]:
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


def chunk_text(text: str, chunk_tokens: int = 550, overlap_tokens: int = 80) -> List[str]:
    """Token-aware chunking when tiktoken exists; otherwise char-based fallback."""
    text = normalize_text(text)
    enc = _get_encoder()

    if enc is None:
        chunk_size = 1800
        overlap = 250
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    tokens = enc.encode(text)
    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_tokens)
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_tokens)

    return chunks


def get_client_and_collection(chroma_dir: str, collection_name: str):
    chromadb, Settings = _get_chromadb()
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def embed_texts(openai_client: OpenAI, embed_model: str, texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in resp.data]


def reset_collection(chroma_dir: str, collection_name: str):
    chromadb, Settings = _get_chromadb()
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def stats(chroma_dir: str, collection_name: str) -> Dict[str, int]:
    _, col = get_client_and_collection(chroma_dir, collection_name)
    return {"count": col.count()}


def _existing_ids(collection, ids: List[str]) -> set[str]:
    """Return set of IDs that already exist in collection (for skip mode)."""
    if not ids:
        return set()
    try:
        got = collection.get(ids=ids, include=[])
        return set(got.get("ids", []) or [])
    except Exception:
        return set()


def ingest(
    data_dir: str,
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
    embed_model: str = "text-embedding-3-small",
    chunk_tokens: int = 550,
    overlap_tokens: int = 80,
    batch_size: int = 64,
    update_mode: str = "update",   # skip | update | replace
):
    """
    update_mode:
      - replace: delete & recreate collection, then ingest
      - update: upsert all chunks (default)
      - skip: only upsert chunks whose IDs do NOT already exist
    """
    update_mode = (update_mode or "update").strip().lower()
    if update_mode not in {"skip", "update", "replace"}:
        raise ValueError("update_mode must be one of: skip, update, replace")

    if update_mode == "replace":
        reset_collection(chroma_dir, collection_name)

    openai_client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=openai_key)
    _, collection = get_client_and_collection(chroma_dir, collection_name)

    docs = iter_documents(data_dir)
    if not docs:
        raise RuntimeError(f"No documents found under {data_dir}. Expected data/<mission>/**/*.txt")

    for mission, fp, raw in docs:
        text = normalize_text(raw)
        chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

        file_name = Path(fp).name
        try:
            category = Path(fp).parent.name
        except Exception:
            category = "document"

        ids = [f"{mission}:{file_name}:{i}" for i in range(len(chunks))]
        metadatas = [{"mission": mission, "source": fp, "category": category, "chunk_id": i}
                     for i in range(len(chunks))]

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            if update_mode == "skip":
                exists = _existing_ids(collection, batch_ids)
                if exists:
                    filtered = [(c, _id, m) for c, _id, m in zip(batch_chunks, batch_ids, batch_meta)
                                if _id not in exists]
                    if not filtered:
                        continue
                    batch_chunks, batch_ids, batch_meta = map(list, zip(*filtered))

            vectors = embed_texts(openai_client, embed_model, batch_chunks)
            collection.upsert(ids=batch_ids, documents=batch_chunks, embeddings=vectors, metadatas=batch_meta)


def main():
    parser = argparse.ArgumentParser(description="NASA Embedding Pipeline -> ChromaDB")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest .txt/.md files into ChromaDB")
    p_ingest.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    p_ingest.add_argument("--chroma-dir", default=os.getenv("CHROMA_DIR", "chroma_db"))
    p_ingest.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "nasa_missions"))
    p_ingest.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", "voc-2009866425126677464339669333e56cd9164.50366416"))
    p_ingest.add_argument("--embed-model", default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    p_ingest.add_argument("--chunk-tokens", type=int, default=int(os.getenv("CHUNK_TOKENS", "550")))
    p_ingest.add_argument("--overlap-tokens", type=int, default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "80")))
    p_ingest.add_argument("--batch-size", type=int, default=64)

    #Fixing update mode
    p_ingest.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="update",
        help="skip: do not overwrite existing IDs; update: upsert (default); replace: drop & rebuild collection"
    )

    p_reset = sub.add_parser("reset", help="Delete and recreate the Chroma collection")
    p_reset.add_argument("--chroma-dir", default=os.getenv("CHROMA_DIR", "chroma_db"))
    p_reset.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "nasa_missions"))

    p_stats = sub.add_parser("stats", help="Show collection statistics")
    p_stats.add_argument("--chroma-dir", default=os.getenv("CHROMA_DIR", "chroma_db"))
    p_stats.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "nasa_missions"))

    args = parser.parse_args()

    if args.command == "reset":
        reset_collection(args.chroma_dir, args.collection)
        print("✅ Collection reset.")
    elif args.command == "stats":
        print(stats(args.chroma_dir, args.collection))
    elif args.command == "ingest":
        if not args.openai_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --openai-key.")
        ingest(
            data_dir=args.data_dir,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            openai_key=args.openai_key,
            embed_model=args.embed_model,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            batch_size=args.batch_size,
            update_mode=args.update_mode,
        )
        print("✅ Ingestion complete.")
        print(stats(args.chroma_dir, args.collection))


if __name__ == "__main__":
    main()