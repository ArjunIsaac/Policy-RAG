"""
vector_store.py
---------------
ChromaDB + LangChain vector store for insurance policy RAG.
Compatible with chromadb >=1.0 and langchain-chroma >=0.2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from ingestor import Chunk

COLLECTION_NAME = "insurance_policies"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_K = 6


def _chunk_id(chunk: Chunk) -> str:
    # Hash the FULL text so chunks that share an 80-char prefix get distinct IDs
    key = f"{chunk.metadata['source']}::p{chunk.metadata['page']}::l{chunk.metadata.get('line', 0)}::{chunk.text}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class PolicyVectorStore:
    """
    Wraps ChromaDB + LangChain Chroma.

    Usage
    -----
    store = PolicyVectorStore()
    store.add_chunks(chunks)
    results = store.retrieve("waiting period?")
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        embed_model: str = EMBED_MODEL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        print(f"[vector_store] Initialising embeddings ({embed_model}) …")
        self.embeddings = OllamaEmbeddings(model=embed_model)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        print("[vector_store] Ready.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        if not chunks:
            return 0

        # Fetch existing IDs
        try:
            col = self._client.get_collection(self.collection_name)
            existing = set(col.get(include=[])["ids"])
        except Exception:
            existing = set()

        texts, metadatas, ids = [], [], []
        seen_this_batch: set[str] = set()   # guard against duplicates within the batch
        for chunk in chunks:
            cid = _chunk_id(chunk)
            if cid in existing or cid in seen_this_batch:
                continue
            seen_this_batch.add(cid)
            meta = {
                **chunk.metadata,
                "tags": json.dumps(chunk.metadata.get("tags", [])),
                "clause": chunk.metadata.get("clause", ""),
            }
            texts.append(chunk.text)
            metadatas.append(meta)
            ids.append(cid)

        if not texts:
            print("[vector_store] All chunks already indexed.")
            return 0

        total = len(texts)
        print(f"[vector_store] Embedding {total} new chunks …")
        for start in range(0, total, batch_size):
            end = start + batch_size
            self._vectorstore.add_texts(
                texts=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            print(f"[vector_store]   {min(end, total)}/{total}")

        print(f"[vector_store] Indexed {total} chunks.")
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        source_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns top-k chunks. source_filter is a list of PDF filenames to
        restrict retrieval to (None = all sources).
        """
        where: dict | None = None
        if source_filter and len(source_filter) == 1:
            where = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"source": {"$in": source_filter}}

        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k, filter=where
        )

        output = []
        for doc, score in results:
            output.append({
                "text": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "line": doc.metadata.get("line", "?"),
                "clause": doc.metadata.get("clause", ""),
                "source": doc.metadata.get("source", "?"),
                "type": doc.metadata.get("type", "text"),
                "tags": json.loads(doc.metadata.get("tags", "[]")),
                "score": round(score, 4),
            })
        return output

    def as_retriever(
        self,
        k: int = DEFAULT_K,
        source_filter: list[str] | None = None,
    ):
        """LangChain-compatible retriever."""
        search_kwargs: dict[str, Any] = {"k": k}
        if source_filter and len(source_filter) == 1:
            search_kwargs["filter"] = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            search_kwargs["filter"] = {"source": {"$in": source_filter}}
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        try:
            col = self._client.get_collection(self.collection_name)
            all_meta = col.get(include=["metadatas"])["metadatas"]
            return sorted({m.get("source", "") for m in all_meta if m})
        except Exception:
            return []

    def count(self) -> int:
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    def delete_source(self, source_name: str) -> None:
        """Remove all chunks belonging to a specific PDF."""
        try:
            col = self._client.get_collection(self.collection_name)
            result = col.get(where={"source": {"$eq": source_name}}, include=[])
            ids = result.get("ids", [])
            if ids:
                col.delete(ids=ids)
                print(f"[vector_store] Deleted {len(ids)} chunks for '{source_name}'")
        except Exception as e:
            print(f"[vector_store] Could not delete '{source_name}': {e}")