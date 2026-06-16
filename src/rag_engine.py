# src/rag_engine.py
# Phase 5 & 6 — Document Learning System + RAG Knowledge Base
# Uploads PDFs/mark schemes/examiner reports → vector DB → retrieved at grading time

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))
from config import KNOWLEDGE_DIR

# ── Vector store (uses ChromaDB — lightweight, no server needed) ──────────────
CHROMA_DIR = KNOWLEDGE_DIR / "chroma_db"
METADATA_FILE = KNOWLEDGE_DIR / "documents.json"


def _get_collection():
    """Returns the ChromaDB collection, creating it if needed."""
    try:
        import chromadb
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name="cambridge_economics",
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except ImportError:
        return None


def _get_embedder():
    """Returns a sentence transformer embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Splits text into overlapping chunks for embedding."""
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]


def _load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(meta: dict):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def add_document(
    file_bytes:    bytes,
    filename:      str,
    doc_type:      str = "general",   # "mark_scheme", "examiner_report", "textbook", "notes"
    topic:         str = "",
) -> dict:
    """
    Adds a document to the knowledge base.
    Extracts text (with OCR fallback), chunks it, embeds, stores in ChromaDB.
    Returns status dict.
    """
    from ocr_pipeline import extract_text_from_bytes

    collection = _get_collection()
    embedder   = _get_embedder()

    if collection is None:
        return {"success": False, "error": "chromadb not installed. Run: pip install chromadb"}
    if embedder is None:
        return {"success": False, "error": "sentence-transformers not installed. Run: pip install sentence-transformers"}

    # Extract text
    text, method = extract_text_from_bytes(file_bytes, filename)
    if not text or len(text.strip()) < 50:
        return {"success": False, "error": f"Could not extract text from {filename} (method: {method})"}

    # Chunk
    chunks    = _chunk_text(text)
    doc_hash  = hashlib.md5(file_bytes).hexdigest()[:12]

    # Check if already indexed
    meta = _load_metadata()
    if doc_hash in meta:
        return {"success": True, "skipped": True, "message": f"{filename} already in knowledge base"}

    # Embed and store
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids        = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas  = [{"filename": filename, "doc_type": doc_type, "topic": topic, "chunk": i}
                  for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

    # Save metadata
    meta[doc_hash] = {
        "filename": filename, "doc_type": doc_type, "topic": topic,
        "chunks": len(chunks), "method": method,
        "word_count": len(text.split()),
    }
    _save_metadata(meta)

    return {
        "success": True, "skipped": False,
        "chunks": len(chunks), "method": method,
        "word_count": len(text.split()),
    }


def retrieve_context(query: str, n_results: int = 4, doc_type_filter: Optional[str] = None) -> str:
    """
    Retrieves the most relevant chunks from the knowledge base for a query.
    Returns a formatted string ready to inject into a grading prompt.
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    if collection is None or embedder is None:
        return ""

    try:
        count = collection.count()
        if count == 0:
            return ""

        query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

        where = {"doc_type": doc_type_filter} if doc_type_filter else None
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            relevance = round((1 - dist) * 100)
            source    = f"{meta.get('doc_type','').upper()}: {meta.get('filename','')} [{relevance}% match]"
            context_parts.append(f"[{source}]\n{doc}")

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"[rag_engine] Retrieval error: {e}")
        return ""


def list_documents() -> list[dict]:
    """Returns all documents in the knowledge base."""
    meta = _load_metadata()
    return [{"hash": k, **v} for k, v in meta.items()]


def delete_document(doc_hash: str) -> bool:
    """Removes a document and its chunks from the knowledge base."""
    collection = _get_collection()
    meta       = _load_metadata()

    if doc_hash not in meta:
        return False

    info    = meta[doc_hash]
    n_chunks = info.get("chunks", 0)
    ids     = [f"{doc_hash}_{i}" for i in range(n_chunks)]

    try:
        collection.delete(ids=ids)
    except Exception as e:
        print(f"[rag_engine] Delete error: {e}")

    del meta[doc_hash]
    _save_metadata(meta)
    return True


def is_rag_available() -> bool:
    """Returns True if RAG dependencies are installed."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False


def get_rag_status() -> dict:
    """Returns RAG system status for the UI."""
    available = is_rag_available()
    docs      = list_documents() if available else []
    collection = _get_collection() if available else None
    chunk_count = collection.count() if collection else 0

    return {
        "available":   available,
        "doc_count":   len(docs),
        "chunk_count": chunk_count,
        "documents":   docs,
    }
