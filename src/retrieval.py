"""
retrieval.py
------------
Hybrid retrieval pipeline:
  - Query expansion / transformation
  - Forced chunk injection for targeted queries
  - Cross-encoder reranking
  - LongContextReorder
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Dict, List

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from constants import (
    CROSS_ENCODER_MODEL,
    FINAL_K,
    HYBRID_FETCH_K,
    REORDER_ENABLED,
    STOPWORDS,
)

if TYPE_CHECKING:
    from vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Query transformation
# ---------------------------------------------------------------------------

def remove_stopwords(text: str) -> str:
    words = text.lower().split()
    return " ".join(w for w in words if w not in STOPWORDS and len(w) > 2)


def transform_query(query: str) -> str:


    q = query
    ql = query.lower()

    synonym_map = {
        "ped": ["pre existing disease", "pre-existing disease"],
        "pre existing": ["pre-existing"],
        "copay": ["co-payment", "co payment"],
        "co-pay": ["co-payment"],
        "copayment": ["co-payment"],
        "ncb": ["no claim bonus"],
        "icu": ["intensive care unit"],
        "opd": ["outpatient", "out patient"],
        "ayush": [
            "ayurveda",
            "yoga",
            "unani",
            "siddha",
            "homeopathy",
        ],
        "maternity": ["pregnancy", "childbirth"],
        "hospitalisation": ["hospitalization"],
        "hospitalization": ["hospitalisation"],
    }

    additions = []

    for key, synonyms in synonym_map.items():
        if key in ql:
            additions.extend(synonyms)

    # Remove duplicates while preserving order
    seen = set()
    additions = [
        s for s in additions
        if not (s.lower() in seen or seen.add(s.lower()))
    ]

    if additions:
        q += " " + " ".join(additions)

    return q


def extract_condition(query: str) -> str | None:


    q = query.lower()

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", q)

    stopwords = {
        "what", "is", "the", "a", "an",
        "for", "of", "on", "under",
        "does", "do",
        "can", "may",
        "please",
    }

    keywords = [w for w in words if w not in stopwords]

    if not keywords:
        return None

    return " ".join(keywords)


# ---------------------------------------------------------------------------
# Forced chunk injection
# ---------------------------------------------------------------------------

def fetch_forced_chunks(
    store: "PolicyVectorStore",
    source_filter: list[str] | None,
    condition: str | None = None,
) -> List[Document]:
    """
    Bypass vector search and scan ChromaDB directly for structurally important chunks.
    Guarantees inclusion of exclusion, critical illness, and waiting period sections.
    """
    try:
        col = store._client.get_collection(store.collection_name)
        where = None
        if source_filter:
            where = (
                {"source": source_filter[0]}
                if len(source_filter) == 1
                else {"source": {"$in": source_filter}}
            )
        result = (
            col.get(where=where, include=["metadatas", "documents"])
            if where
            else col.get(include=["metadatas", "documents"])
        )

        search_terms = []
        if condition:
            search_terms = [t.strip() for t in re.split(r"[\s\-]+", condition) if len(t.strip()) > 3]

        docs: List[Document] = []
        for meta, text in zip(result["metadatas"], result["documents"]):
            heading   = (meta.get("heading") or "").lower()
            text_lower = text.lower()
            match = False

            if search_terms and all(t in text_lower for t in search_terms):
                match = True
            elif condition and condition in text_lower:
                match = True
            if "critical illness" in heading or "critical illness" in text_lower:
                match = True
            if "exclusion" in heading or "exclusion" in text_lower:
                match = True
            if "waiting period" in heading or "waiting period" in text_lower:
                match = True
            if any(k in heading or k in text_lower for k in [
                "cancer", "carcinoma", "melanoma", "eye surgery",
                "cataract", "lasik", "refractive", "dioptres",
            ]):
                match = True

            if match:
                parent = meta.get("parent_text", text)
                docs.append(Document(
                    page_content=parent,
                    metadata={
                        "source":      meta.get("source"),
                        "page":        meta.get("page"),
                        "heading":     meta.get("heading", ""),
                        "clause":      meta.get("clause", ""),
                        "parent_text": parent,
                    },
                ))

        # Deduplicate by (page, heading)
        seen: set = set()
        unique: List[Document] = []
        for doc in docs:
            key = (doc.metadata.get("page"), doc.metadata.get("heading"))
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique[:8]
    except Exception as e:
        print(f"[retrieval] fetch_forced_chunks error: {e}")
        return []


# ---------------------------------------------------------------------------
# Cross-encoder (lazy singleton)
# ---------------------------------------------------------------------------

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("[retrieval] Loading CrossEncoder…")
        sys.stdout.flush()
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Main retrieval pipeline
# ---------------------------------------------------------------------------

def retrieve_docs(
    store: "PolicyVectorStore",
    query: str,
    source_filter: list[str] | None,
    k: int = FINAL_K,
    debug: bool = False,
) -> List[Document]:
    """
    Full retrieval pipeline:
    1. Hybrid (vector + BM25) search
    2. Optional forced chunk injection for targeted medical queries
    3. Parent-text expansion + deduplication
    4. Cross-encoder reranking
    5. LongContextReorder
    6. Truncate to k
    """
    ql = query.lower()
    is_targeted = any(t in ql for t in [
        "cancer", "carcinoma", "melanoma", "tumor",
        "eye", "surgery", "cataract", "lasik", "refractive",
    ])
    search_type = "similarity"

    # 1. Hybrid retrieval
    hybrid = store.as_hybrid_retriever(
        k=HYBRID_FETCH_K,
        source_filter=source_filter,
        search_type=search_type,
    )
    raw_docs: List[Document] = hybrid.invoke(query) or []

    # 2. Forced chunks for targeted queries
    forced_docs: List[Document] = []
    ENABLE_FORCED_RETRIEVAL = False # FLAG FOR BENCHMARK
    if is_targeted and ENABLE_FORCED_RETRIEVAL:

        print("[retrieval] Targeted query — injecting forced chunks")
        sys.stdout.flush()
        condition = extract_condition(query)

        forced_docs = fetch_forced_chunks(store, source_filter, condition)

    # 3. Cross-encoder rerank ALL retrieved child chunks
    cross_enc = get_cross_encoder()

    if raw_docs:
        pairs = []

        for doc in raw_docs:
            text = doc.page_content

            heading = doc.metadata.get("heading", "")
            clause = doc.metadata.get("clause", "")

            # Give the reranker structural context
            if heading:
                text = f"Heading: {heading}\n\n{text}"

            if clause and clause != heading:
                text = f"Clause: {clause}\n\n{text}"

            pairs.append((query, text))

        scores = cross_enc.predict(pairs)

        scored_docs = sorted(
            zip(raw_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

    else:
        scored_docs = []

    # 4. Deduplicate AFTER reranking
    parent_map: Dict[str, Document] = {}

    for doc, _ in scored_docs:
        parent = doc.metadata.get("parent_text", doc.page_content)

        if parent not in parent_map:
            parent_map[parent] = doc

    reranked = list(parent_map.values())

    if debug:
        print(f"[retrieval] Reranked {len(raw_docs)} child chunks")
        print(f"[retrieval] Kept {len(reranked)} unique parents")
        sys.stdout.flush()

    # 5. Merge: forced first, then reranked (no duplicates)
    forced_keys = {(d.metadata.get("page"), d.metadata.get("heading")) for d in forced_docs}
    final: List[Document] = list(forced_docs)
    for doc in reranked:
        key = (doc.metadata.get("page"), doc.metadata.get("heading"))
        if key not in forced_keys:
            final.append(doc)

    # 6. Reorder + truncate
    if REORDER_ENABLED and final:
        final = LongContextReorder().transform_documents(final)
    final = final[:k]

    if debug:
        print(f"[retrieval] Final {len(final)} docs")
        for i, doc in enumerate(final):
            heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
            print(f"  {i+1}: Page {doc.metadata.get('page','?')} | {heading}")
            print(f"      {doc.page_content[:200]}...")
        print("=" * 60)
        sys.stdout.flush()

    return final