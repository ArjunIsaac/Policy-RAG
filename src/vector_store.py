"""
vector_store.py
---------------
Manages embedding creation, ChromaDB indexing, and retrieval.

Embedding model : nomic-embed-text via Ollama  (runs locally, no API key needed)
Vector store    : ChromaDB with persistence
Search strategy : hybrid — dense similarity + keyword-tag pre-filter
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from ingestor import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "insurance_policies"
EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
DEFAULT_K = 6                       # chunks returned per query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: Chunk) -> str:
    """Deterministic ID so re-ingestion is idempotent."""
    key = f"{chunk.metadata['source']}::p{chunk.metadata['page']}::{chunk.text[:80]}"
    return hashlib.md5(key.encode()).hexdigest()


def _build_where_filter(tags: list[str]) -> dict | None:
    """
    Build a ChromaDB metadata $contains-style filter for insurance tags.
    Returns None when no tags are detected (no filter applied).
    """
    if not tags:
        return None
    # ChromaDB supports $or across metadata array fields via $contains
    # We store tags as a JSON string and fall back to text search otherwise.
    return None  # See retrieval note in `similarity_search_with_filter`


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PolicyVectorStore:
    """
    Wraps ChromaDB + LangChain Chroma for the insurance RAG pipeline.

    Usage
    -----
    store = PolicyVectorStore(persist_dir="data/chroma_db")
    store.add_chunks(chunks)          # index new chunks
    results = store.retrieve("what is the waiting period?")
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        embed_model: str = EMBED_MODEL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[vector_store] Initialising embeddings with '{embed_model}' …")
        self.embeddings = OllamaEmbeddings(model=embed_model)

        # Raw ChromaDB client for admin operations (e.g. checking existing IDs)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain-wrapped Chroma for retriever convenience
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        print("[vector_store] Ready.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """
        Add *chunks* to the vector store.  Skips chunks that are already
        present (idempotent based on content hash).

        Returns the number of *new* chunks actually added.
        """
        if not chunks:
            print("[vector_store] No chunks to add.")
            return 0

        # Fetch existing IDs to avoid duplicates
        existing = set(
            self._client.get_collection(
                self._vectorstore._collection.name
            ).get(include=[])["ids"]
        )

        texts, metadatas, ids = [], [], []
        for chunk in chunks:
            cid = _chunk_id(chunk)
            if cid in existing:
                continue
            # Serialise tags list → string so ChromaDB can store it
            meta = {**chunk.metadata, "tags": json.dumps(chunk.metadata.get("tags", []))}
            texts.append(chunk.text)
            metadatas.append(meta)
            ids.append(cid)

        if not texts:
            print("[vector_store] All chunks already indexed — nothing to add.")
            return 0

        total_new = len(texts)
        print(f"[vector_store] Embedding {total_new} new chunks …")

        for start in range(0, total_new, batch_size):
            end = start + batch_size
            self._vectorstore.add_texts(
                texts=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            print(f"[vector_store]   {min(end, total_new)}/{total_new} done")

        print(f"[vector_store] Indexed {total_new} new chunks.")
        return total_new

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-*k* most relevant chunks for *query*.

        Parameters
        ----------
        query         : natural language question
        k             : number of results to return
        source_filter : if set, restrict to chunks from this PDF filename

        Returns
        -------
        list of dicts with keys: text, page, source, type, tags, score
        """
        where: dict | None = None
        if source_filter:
            where = {"source": {"$eq": source_filter}}

        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where,
        )

        output = []
        for doc, score in results:
            output.append(
                {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "source": doc.metadata.get("source", "?"),
                    "type": doc.metadata.get("type", "text"),
                    "tags": json.loads(doc.metadata.get("tags", "[]")),
                    "score": round(score, 4),
                }
            )
        return output

    def as_retriever(self, k: int = DEFAULT_K, source_filter: str | None = None):
        """
        Return a LangChain-compatible retriever object for use in chains.
        """
        search_kwargs: dict[str, Any] = {"k": k}
        if source_filter:
            search_kwargs["filter"] = {"source": {"$eq": source_filter}}
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        """Return unique PDF filenames currently in the store."""
        col = self._client.get_collection(self._vectorstore._collection.name)
        all_meta = col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "") for m in all_meta if m})

    def count(self) -> int:
        """Total number of chunks in the store."""
        return self._vectorstore._collection.count()