"""
vector_store.py
---------------
ChromaDB + LangChain vector store for insurance policy RAG.
Now supports hybrid search with configurable search type and BM25 weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma


from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from datatypes import Chunk

COLLECTION_NAME = "insurance_policies"
# Updated to the official Hugging Face repo ID
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
HYBRID_FETCH_K = 20  

def _chunk_id(chunk: Chunk) -> str:
    key = (
        f"{chunk.metadata['source']}::p{chunk.metadata['page']}"
        f"::h{chunk.metadata.get('heading', '')}::{chunk.text}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:32]

class ONNXSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(
            model_name,
            backend="onnx",
            device="cpu",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()
    
    
class PolicyVectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        embed_model: str = EMBED_MODEL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        print(f"[vector_store] Initialising embeddings ({embed_model}) on CPU…")
        # Swapped OllamaEmbeddings for HuggingFaceEmbeddings
        # Forced to CPU so it does not steal VRAM from your vLLM server
        self.embeddings = ONNXSentenceTransformerEmbeddings(embed_model)

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

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        if not chunks:
            return 0

        try:
            col = self._client.get_collection(self.collection_name)
            existing = set(col.get(include=[])["ids"])
        except Exception:
            existing = set()

        texts, metadatas, ids = [], [], []
        seen_this_batch: set[str] = set()

        for chunk in chunks:
            cid = _chunk_id(chunk)
            if cid in existing or cid in seen_this_batch:
                continue
            seen_this_batch.add(cid)
            meta = {
                **chunk.metadata,
                "tags": json.dumps(chunk.metadata.get("tags", [])),
                "clause": chunk.metadata.get("clause", ""),
                "heading": chunk.metadata.get("heading", ""),
                "parent_text": chunk.metadata.get("parent_text", chunk.text),
            }
            texts.append(chunk.metadata.get("embedding_text", chunk.text))
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

    def as_hybrid_retriever(
        self,
        k: int = HYBRID_FETCH_K,
        source_filter: list[str] | None = None,
        search_type: str = "mmr"
    ):
        """Return a hybrid retriever (vector + BM25)."""
        search_kwargs: dict[str, Any] = {"k": k}
        where = None
        
        if source_filter and len(source_filter) == 1:
            where = {"policy_id": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"policy_id": {"$in": source_filter}}
            
        if where:
            search_kwargs["filter"] = where

        # 1. Vector Retriever
        if search_type == "mmr":
            vector_retriever = self._vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={**search_kwargs, "fetch_k": k * 2, "lambda_mult": 0.8}
            )
        else:
            vector_retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )

        # 2. BM25 Retriever
        try:
            col = self._client.get_collection(self.collection_name)
            chroma_data = col.get(where=where) if where else col.get()
            
            docs = []
            for text, meta in zip(chroma_data['documents'], chroma_data['metadatas']):
                docs.append(Document(page_content=text, metadata=meta))
                
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = k
                
                # 3. Combine them – favour BM25 for exact matches
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever], 
                    weights=[0.6, 0.4]  # Adjust weights as needed
                )
                return ensemble_retriever
            else:
                print("[vector_store] No documents for BM25, using vector only")
                return vector_retriever
        except Exception as e:
            print(f"[vector_store] BM25 setup failed, falling back to Vector. Error: {e}")
            return vector_retriever

    def retrieve(self, query: str, k: int = 10, source_filter: list[str] | None = None) -> list[dict[str, Any]]:
        where: dict | None = None
        
        # FIX: Use "policy_id" not "source" in the filter
        if source_filter:
            if len(source_filter) == 1:
                where = {"policy_id": {"$eq": source_filter[0]}}
                print(f"[vector_store] Filtering by policy: {source_filter[0]}")
            elif len(source_filter) > 1:
                where = {"policy_id": {"$in": source_filter}}
                print(f"[vector_store] Filtering by policies: {source_filter}")

        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query, 
            k=k, 
            filter=where
        )
        
        # Log which policies were retrieved
        policies_seen = {}
        for doc, score in results:
            pid = doc.metadata.get("policy_id", "UNKNOWN")
            policies_seen[pid] = policies_seen.get(pid, 0) + 1
        print(f"[vector_store] Retrieved from policies: {policies_seen}")
        
        output = []
        for doc, score in results:
            output.append({
                "text": doc.page_content,
                "parent_text": doc.metadata.get("parent_text", doc.page_content),
                "page": doc.metadata.get("page", "?"),
                "line": doc.metadata.get("line", "?"),
                "clause": doc.metadata.get("clause", ""),
                "heading": doc.metadata.get("heading", ""),
                "source": doc.metadata.get("source", "?"),
                "policy_id": doc.metadata.get("policy_id", "UNKNOWN"),  # Add this
                "score": round(score, 4),
            })
        return output

    def list_sources(self) -> list[str]:
        try:
            col = self._client.get_collection(self.collection_name)
            all_meta = col.get(include=["metadatas"])["metadatas"]
            return sorted({m.get("policy_id", "") for m in all_meta if m and m.get("policy_id")})
        except Exception:
            return []

    def count(self) -> int:
        try:
            return self._client.get_collection(self.collection_name).count()
        except Exception:
            return 0
        

    def verify_policies(self) -> dict:
        try:
            col = self._client.get_collection(self.collection_name)
            all_meta = col.get(include=["metadatas"])["metadatas"]
            
            policy_ids = {}
            for meta in all_meta:
                pid = meta.get("policy_id", "UNKNOWN")
                if pid not in policy_ids:
                    policy_ids[pid] = 0
                policy_ids[pid] += 1
            
            print(f"[vector_store] Found policies in Chroma: {policy_ids}")
            return policy_ids
        except Exception as e:
            print(f"[vector_store] Error verifying: {e}")
            return {}