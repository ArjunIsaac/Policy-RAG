"""
retrieval_engine.py - Smart retrieval and chunk routing
Retrieves top chunks, scores against attributes, dynamically allocates
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from .config import CRITICAL_ATTRIBUTES, RELEVANCE_KEYWORDS, AttributeConfig

if TYPE_CHECKING:
    from vector_store import PolicyVectorStore


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""
    content: str
    page: int
    chunk_id: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartRetriever:
    
    def __init__(self, store: "PolicyVectorStore", top_k: int = 20):
        self.store = store
        self.top_k = top_k
    
    def retrieve_and_route(self, source_filter: List[str] = None) -> Dict[str, List[RetrievedChunk]]:
        # Step 1: Build master query
        master_query = self._build_master_query()
        
        # Step 2: Retrieve top chunks
        all_chunks = self._retrieve_chunks(master_query, source_filter)
        
        if not all_chunks:
            print("[SmartRetriever] No chunks retrieved - check vector store")
            return {}
        
        print(f"[SmartRetriever] Retrieved {len(all_chunks)} chunks")
        
        # Step 3: Score chunks against attributes
        attribute_chunks = self._score_and_route(all_chunks)
        
        # Step 4: Dynamic allocation per attribute
        final_allocation = self._allocate_chunks(attribute_chunks)
        
        return final_allocation
    
    def _build_master_query(self) -> str:
        """Build a comprehensive query covering all attributes."""
        parts = []
        for config in CRITICAL_ATTRIBUTES.values():
            parts.append(config.question)
            for alt in config.alternatives[:2]:
                parts.append(alt)
        
        # Limit query length to avoid token issues
        query = " ".join(parts[:30])
        return query
    
    def _retrieve_chunks(self, query: str, source_filter: List[str]) -> List[RetrievedChunk]:
        """Retrieve chunks from vector store."""
        # Build filter
        filter_dict = None
        if source_filter and len(source_filter) == 1:
            filter_dict = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            filter_dict = {"source": {"$in": source_filter}}
        
        # PRIMARY: Use similarity search
        try:
            if hasattr(self.store, 'similarity_search_with_score'):
                results = self.store.similarity_search_with_score(
                    query, 
                    k=self.top_k,
                    filter=filter_dict
                )
                
                if results:
                    chunks = []
                    for doc, score in results:
                        chunks.append(RetrievedChunk(
                            content=doc.page_content,
                            page=doc.metadata.get("page", 0),
                            chunk_id=doc.metadata.get("chunk_id", str(id(doc))),
                            similarity_score=score,
                            metadata=doc.metadata
                        ))
                    print(f"[SmartRetriever] Retrieved {len(chunks)} chunks via similarity search")
                    return chunks
                    
        except Exception as e:
            print(f"[SmartRetriever] Similarity search error: {e}")
        
        # FALLBACK 1: Try using the vector store's retriever
        try:
            retriever = self.store.as_hybrid_retriever(
                k=self.top_k,
                source_filter=source_filter
            )
            docs = retriever.invoke(query)
            
            if docs:
                chunks = []
                for doc in docs:
                    chunks.append(RetrievedChunk(
                        content=doc.page_content,
                        page=doc.metadata.get("page", 0),
                        chunk_id=doc.metadata.get("chunk_id", str(id(doc))),
                        similarity_score=0.5,
                        metadata=doc.metadata
                    ))
                print(f"[SmartRetriever] Retrieved {len(chunks)} chunks via hybrid retriever")
                return chunks
                
        except Exception as e:
            print(f"[SmartRetriever] Hybrid retriever error: {e}")
        
        # FALLBACK 2: Direct ChromaDB access
        try:
            col = self.store._client.get_collection(self.store.collection_name)
            
            result = col.get(
                where=filter_dict if filter_dict else None,
                include=["documents", "metadatas"]
            )
            
            if result and result["documents"]:
                chunks = []
                for text, meta in zip(result["documents"], result["metadatas"]):
                    chunks.append(RetrievedChunk(
                        content=text,
                        page=meta.get("page", 0),
                        chunk_id=meta.get("chunk_id", str(id(text))),
                        similarity_score=0.5,
                        metadata=meta
                    ))
                print(f"[SmartRetriever] Retrieved {len(chunks)} chunks via direct ChromaDB access")
                return chunks[:self.top_k]
                
        except Exception as e:
            print(f"[SmartRetriever] Direct ChromaDB access error: {e}")
        
        # FALLBACK 3: Try to get all text and chunk manually
        try:
            # Try to get all documents from the store
            col = self.store._client.get_collection(self.store.collection_name)
            result = col.get(include=["documents", "metadatas"])
            
            if result and result["documents"]:
                # Sort by page
                pairs = sorted(
                    zip(result["documents"], result["metadatas"]),
                    key=lambda x: x[1].get("page", 0)
                )
                
                # Combine all text
                full_text = "\n\n".join(pairs[0][0] for pairs in pairs)
                
                # Simple chunking by paragraphs
                paragraphs = re.split(r'\n\s*\n', full_text)
                chunks = []
                for i, para in enumerate(paragraphs[:self.top_k]):
                    if len(para.strip()) > 50:
                        chunks.append(RetrievedChunk(
                            content=para.strip(),
                            page=1,
                            chunk_id=f"fallback_{i}",
                            similarity_score=0.3,
                            metadata={"source": "fallback"}
                        ))
                
                print(f"[SmartRetriever] Retrieved {len(chunks)} chunks via fallback chunking")
                return chunks
                
        except Exception as e:
            print(f"[SmartRetriever] Final fallback error: {e}")
        
        return []
    
    def _score_and_route(self, chunks: List[RetrievedChunk]) -> Dict[str, List[Dict]]:
        """
        Score each chunk against all attributes.
        Returns: attribute_name → list of {chunk, score}
        """
        attribute_chunks = defaultdict(list)
        
        for chunk in chunks:
            content = chunk.content.lower()
            
            for attr_name, config in CRITICAL_ATTRIBUTES.items():
                score = self._calculate_relevance_score(content, attr_name, config)
                
                if score > 0.2:  # Minimum relevance threshold
                    attribute_chunks[attr_name].append({
                        "chunk": chunk,
                        "score": score,
                        "content": content
                    })
        
        return dict(attribute_chunks)
    
    def _calculate_relevance_score(self, content: str, attr_name: str, config: AttributeConfig) -> float:

        score = 0.0
        
        # Factor 1: Exact attribute name match (30% weight)
        if attr_name.replace("_", " ") in content:
            score += 0.30
        
        # Factor 2: Keywords from config (40% weight)
        keywords = RELEVANCE_KEYWORDS.get(attr_name, [])
        keyword_matches = sum(1 for kw in keywords if kw in content)
        if keywords:
            keyword_score = (keyword_matches / len(keywords)) * 0.40
            score += keyword_score
        
        # Factor 3: Alternative phrasings (20% weight)
        alt_matches = sum(1 for alt in config.alternatives if alt.lower() in content)
        if config.alternatives:
            alt_score = (alt_matches / len(config.alternatives)) * 0.20
            score += alt_score
        
        # Factor 4: Structural indicators (10% weight)
        if re.search(r'\d+', content):
            score += 0.05  # Contains numbers
        if re.search(r'(?:section|clause|schedule|table)', content):
            score += 0.05  # Structural
        
        return min(1.0, score)
    
    def _allocate_chunks(self, attribute_chunks: Dict[str, List[Dict]]) -> Dict[str, List[RetrievedChunk]]:
        """
        Dynamically allocate chunks per attribute.
        Rules:
            - Minimum: 1 chunk per attribute (if available)
            - Maximum: 3 chunks per attribute
            - Only allocate second chunk if score > 0.5
            - Only allocate third chunk if score > 0.7
        """
        final_allocation = {}
        
        for attr_name, chunks_with_scores in attribute_chunks.items():
            sorted_chunks = sorted(chunks_with_scores, key=lambda x: x["score"], reverse=True)
            
            if not sorted_chunks:
                final_allocation[attr_name] = []
                continue
            
            allocated = []
            
            # Always include the best chunk
            allocated.append(sorted_chunks[0]["chunk"])
            
            # Add second chunk if score > 0.5 and exists
            if len(sorted_chunks) > 1 and sorted_chunks[1]["score"] > 0.5:
                allocated.append(sorted_chunks[1]["chunk"])
            
            # Add third chunk if score > 0.7 and exists
            if len(sorted_chunks) > 2 and sorted_chunks[2]["score"] > 0.7:
                allocated.append(sorted_chunks[2]["chunk"])
            
            final_allocation[attr_name] = allocated
        
        # Ensure every attribute has at least some chunks
        for attr_name in CRITICAL_ATTRIBUTES:
            if attr_name not in final_allocation or not final_allocation[attr_name]:
                # Use the first chunk from any attribute
                for chunks in final_allocation.values():
                    if chunks:
                        final_allocation[attr_name] = chunks[:1]
                        break
        
        # Log allocation stats
        total_chunks = sum(len(chunks) for chunks in final_allocation.values())
        print(f"[SmartRetriever] Allocated {total_chunks} chunks across {len(final_allocation)} attributes")
        
        return final_allocation


__all__ = ["SmartRetriever", "RetrievedChunk"]