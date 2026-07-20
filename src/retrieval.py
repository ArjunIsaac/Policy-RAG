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
    RRF_ENABLED,
)

if TYPE_CHECKING:
    from vector_store import PolicyVectorStore

# Helper functions

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


def adaptive_final_k(query : str) -> int:
    # Determine the final number of fetched docs based on query enumeration
    q = query.lower()
    enumeration_words = {'list', 'show', 'all', 'every','each','compare','different','various', 'several'}
    q_tokens = set(q.replace("/"," ").split())
    
    if enumeration_words & q_tokens:
        return 8
    else:
        return FINAL_K



# Forced chunk injection

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


# Cross-encoder (lazy singleton)

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("[retrieval] Loading CrossEncoder…")
        sys.stdout.flush()
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    return _cross_encoder


def build_reranker_text(doc: Document) -> str:
    """
    Build a semantically rich passage for the cross-encoder.
    This text is ONLY used for reranking.
    It is never shown to the LLM.
    """

    md = doc.metadata

    parts = []

    heading = md.get("heading")
    if heading:
        parts.append(f"Section: {heading}")

    clause = md.get("clause")
    if clause and clause != heading:
        parts.append(f"Clause: {clause}")

    chunk_type = md.get("type")
    if chunk_type:
        parts.append(f"Content Type: {chunk_type}")

    tags = md.get("tags")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    if md.get("is_definition"):
        parts.append("Contains: Definition")

    metadata = doc.metadata

    chunk_type = metadata.get("type", "")

    if chunk_type.startswith("table"):
        passage = metadata.get("embedding_text", doc.page_content)
    else:
        passage = doc.page_content

    parts.append("Passage:")
    parts.append(passage)

    return "\n\n".join(parts)

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
    1. Hybrid (vector + BM25) search — scoped PER POLICY when multiple
       policies are active, so no single policy can crowd out another
       at the fetch stage.
    2. Optional forced chunk injection for targeted medical queries
    3. Parent-text expansion + deduplication
    4. Cross-encoder reranking (single pass over the combined pool)
    5. LongContextReorder
    6. Truncate PER POLICY to guarantee representation in the final answer
    """
    ql = query.lower()
    is_targeted = any(t in ql for t in [
        "cancer", "carcinoma", "melanoma", "tumor",
        "eye", "surgery", "cataract", "lasik", "refractive",
    ])
    search_type = "similarity"

    # ------------------------------------------------------------------
    # 1. Hybrid retrieval — PER POLICY fetch
    # ------------------------------------------------------------------
    # source_filter may contain multiple policy_ids. Instead of one pooled
    # call (which lets one policy dominate HYBRID_FETCH_K), fetch each
    # policy's own candidates separately and merge. This guarantees every
    # active policy reaches the reranking stage with real candidates.
    policies = source_filter if source_filter else [None]  # None = no filter (single/all-doc mode)

    raw_docs: List[Document] = []
    for pid in policies:
        pid_filter = [pid] if pid else None
        hybrid = store.as_hybrid_retriever(
            k=HYBRID_FETCH_K,
            source_filter=pid_filter,
            search_type=search_type,
        )
        policy_docs = hybrid.invoke(query) or []
        raw_docs.extend(policy_docs)

    hybrid_rank = {
        id(doc): rank
        for rank, doc in enumerate(raw_docs, start=1)
    }

    if debug:
        print("\n=== Hybrid Retrieval (per-policy fetch) ===")
        for pid in policies:
            count = sum(1 for d in raw_docs if d.metadata.get("policy_id") == pid)
            print(f"  policy_id={pid}: {count} candidates fetched")
        for i, doc in enumerate(raw_docs[:15]):
            heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
            print(
                f"{i+1:2d}. policy_id={doc.metadata.get('policy_id','?')} | "
                f"Page {doc.metadata.get('page','?')} | {heading}"
            )
        print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Forced chunks for targeted queries
    # ------------------------------------------------------------------
    forced_docs: List[Document] = []
    ENABLE_FORCED_RETRIEVAL = False  # FLAG FOR BENCHMARK
    if is_targeted and ENABLE_FORCED_RETRIEVAL:
        print("[retrieval] Targeted query — injecting forced chunks")
        sys.stdout.flush()
        condition = extract_condition(query)
        # NOTE: fetch_forced_chunks also needs per-policy scoping if
        # source_filter has >1 policy — see note below.
        forced_docs = fetch_forced_chunks(store, source_filter, condition)

    # ------------------------------------------------------------------
    # 3. Cross-encoder rerank — single pass over the whole combined pool
    # ------------------------------------------------------------------
    cross_enc = get_cross_encoder()

    if raw_docs:
        pairs = [(query, build_reranker_text(doc)) for doc in raw_docs]
        scores = cross_enc.predict(pairs)

        ce_rank = {
            id(doc): rank
            for rank, (doc, _) in enumerate(sorted(
                zip(raw_docs, scores), key=lambda x: x[1], reverse=True
            ), start=1)
        }
        ce_score_lookup = {id(doc): score for doc, score in zip(raw_docs, scores)}

        if RRF_ENABLED:
            RRF_K = 20
            rrf_scores = []
            for doc in raw_docs:
                h = hybrid_rank.get(id(doc), len(raw_docs) + 1)
                c = ce_rank.get(id(doc), len(raw_docs) + 1)
                score = 1 / (RRF_K + h) + 1 / (RRF_K + c)
                rrf_scores.append((doc, score))

            scored_docs = sorted(
                rrf_scores,
                key=lambda x: (x[1], ce_score_lookup[id(x[0])]),
                reverse=True,
            )
            if debug:
                print("\n=== RRF Fusion (Hybrid + Cross-Encoder) ===")
        else:
            scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
            if debug:
                print("\n=== Cross-Encoder Only (No RRF) ===")

        if debug:
            for i, (doc, score) in enumerate(scored_docs[:15]):
                heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
                print(
                    f"{i+1:2d}. Score={score:.4f} | policy_id={doc.metadata.get('policy_id','?')} | "
                    f"Page {doc.metadata.get('page','?')} | {heading}"
                )
            print("=" * 60)
    else:
        scored_docs = []

    # ------------------------------------------------------------------
    # 4. Deduplicate AFTER reranking (unchanged, but now works across
    #    the multi-policy pool — parent_text dedup is content-based so
    #    it stays policy-agnostic automatically)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Merge: forced first, then reranked (no duplicates)
    # ------------------------------------------------------------------
    forced_keys = {(d.metadata.get("page"), d.metadata.get("heading")) for d in forced_docs}
    combined: List[Document] = list(forced_docs)
    for doc in reranked:
        key = (doc.metadata.get("page"), doc.metadata.get("heading"))
        if key not in forced_keys:
            combined.append(doc)

    # ------------------------------------------------------------------
    # 6. Reorder, then truncate PER POLICY (this is the actual final_k fix)
    # ------------------------------------------------------------------
    if REORDER_ENABLED and combined:
        combined = LongContextReorder().transform_documents(combined)

    per_policy_k = adaptive_final_k(query)
    final = truncate_per_policy(combined, policies, per_policy_k)

    if debug:
        print(f"[retrieval] Final {len(final)} docs (per_policy_k={per_policy_k})")
        for i, doc in enumerate(final):
            heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
            print(f"  {i+1}: policy_id={doc.metadata.get('policy_id','?')} | Page {doc.metadata.get('page','?')} | {heading}")
            print(f"      {doc.page_content[:200]}...")
        print("=" * 60)
        sys.stdout.flush()

    return final


def truncate_per_policy(
    docs: List[Document],
    policies: List[str | None],
    per_policy_k: int,
    max_total_chunks: int = 40,
) -> List[Document]:
    """
    Given a relevance-ranked list of docs spanning multiple policies,
    keep the top `per_policy_k` chunks FOR EACH policy, preserving their
    relative rank order within each policy. Policy-agnostic — makes no
    assumption about which policies are active or how many.

    This is what actually prevents one policy's chunks from crowding out
    another's in the final context sent to the LLM.
    """
    n_policies = max(len(policies), 1)
    if per_policy_k * n_policies > max_total_chunks:
        per_policy_k = max(1, max_total_chunks // n_policies)

    counts: Dict[str | None, int] = {}
    final: List[Document] = []

    for doc in docs:  # docs is already ranked (reranked/RRF order preserved)
        pid = doc.metadata.get("policy_id")
        if counts.get(pid, 0) >= per_policy_k:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        final.append(doc)

    return final