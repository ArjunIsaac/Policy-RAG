"""
chain.py
--------
LCEL-based RAG chain — Mistral via Ollama.
Upgraded to reliably capture table values, handle conditional exclusions,
and robustly isolate complex medical terminology.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, List, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_community.document_transformers import LongContextReorder
from sentence_transformers import CrossEncoder

from vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_FETCH_K = 150
FINAL_K = 12
REORDER_ENABLED = True

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert insurance policy analyst reviewing regulatory policy contracts.

STRICT INSTRUCTIONAL RULES:
1. Answer the question directly using ONLY the RETRIEVED CONTEXT below. Do not assume, infer, or extrapolate.
2. INTERPRET TABLES ACCURATELY: Insurance policies utilize benefit grids. If a policy benefit, surgery type, or clause is associated with the term "NIL", "No Coverage", "0", or "-" inside a table row or text block, this means coverage for that item is completely ZERO / NOT COVERED. You must state this explicitly.
3. If a condition or coverage is subject to a conditional exclusion (e.g., "Excluded unless X happens" or "Covered only if Y is met"), you MUST state that exact condition clearly instead of stating that the policy is unclear or does not mention it.
4. Quote exact policy text or table entries when stating inclusions, exclusions, or conditional requirements.
5. If a condition is definitively and permanently excluded without exception, state: "The policy explicitly excludes...".
6. If a condition is completely unmentioned anywhere in the text, state: "The policy does not mention this condition."
7. Every assertion must be cited exactly in the format: (Page X, Clause: Y) or (Page X, Table: Y) as provided in the context metadata.

RETRIEVED CONTEXT:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Attribute extraction groups
# ---------------------------------------------------------------------------

_EXTRACT_GROUPS = [
    (["policy_name", "insurer", "sum_insured", "renewal_type", "network_hospitals"],
     "policy name insurer sum insured options network hospitals renewal"),
    (["waiting_period_initial", "waiting_period_ped", "waiting_period_specific"],
     "waiting period initial pre-existing disease specific illness days"),
    (["copay_percentage", "copay_conditions"],
     "co-payment co-pay percentage non-network hospital conditions"),
    (["room_rent_sublimit", "icu_sublimit"],
     "room rent sub-limit per day ICU charges"),
    (["maternity_covered", "daycare_procedures", "ncb_benefit", "grace_period_days"],
     "maternity day care no claim bonus NCB grace period renewal"),
    (["exclusions_permanent"],
     "permanent exclusions not covered diseases conditions"),
]

_EXTRACT_SYSTEM = """You are an insurance data extractor. Extract ONLY the \
requested fields from the context. Return valid JSON with no markdown fences. \
Use null for fields not found.

Fields: {fields}
Context: {context}
JSON:"""

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*\n```$", "", raw)  # Fixed: escaped newline and closed string
    return raw.strip()

# ---------------------------------------------------------------------------
# Main Chain
# ---------------------------------------------------------------------------

class PolicyChain:
    def __init__(
        self,
        vector_store: PolicyVectorStore,
        model: str = "mistral",
        temperature: float = 0.0,
        k_docs: int = FINAL_K,
        memory_window: int = 4,
        source_filter: list[str] | None = None,
    ) -> None:
        self._store = vector_store
        self._k = k_docs
        self._source_filter = source_filter
        self._window = memory_window
        self._history: list[HumanMessage | AIMessage] = []

        self._llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=2048,
        )
        self._parser = StrOutputParser()
        self._cross_encoder = None

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            print("[chain] Loading CrossEncoder…")
            sys.stdout.flush()
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _transform_query(self, query: str) -> str:
        q_lower = query.lower()
        if any(term in q_lower for term in ["cancer", "carcinoma", "melanoma", "tumor"]):
            query += " critical illness definition"
        if any(term in q_lower for term in ["excluded", "exclusion", "not covered"]):
            query += " permanent exclusion"
        if any(term in q_lower for term in ["waiting", "period"]):
            query += " waiting period pre-existing"
        if any(term in q_lower for term in ["copay", "co-pay"]):
            query += " co-payment"
        if any(term in q_lower for term in ["eye", "cataract", "lasik", "surgery", "surgeries"]):
            query += " limit sublimit cap NIL table benefit"
        return query

    # ------------------------------------------------------------------
    # Extract condition from query (Punctuation-agnostic)
    # ------------------------------------------------------------------

    def _extract_condition(self, query: str) -> str | None:
        normalized = query.replace('–', '-').replace('—', '-').lower()
        keywords = ["cancer", "carcinoma", "melanoma", "tumor", "eye"]
        
        for kw in keywords:
            if kw in normalized:
                match = re.search(r'([a-z0-9\-\s]+?\b' + kw + r'\b)', normalized)
                if match:
                    return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Direct database filtering for forced chunks
    # ------------------------------------------------------------------

    def _fetch_forced_chunks(self, condition: str | None = None) -> List[Document]:
        """
        Fetch chunks programmatically to bypass un-indexed string matching limitations 
        and surface exact phrase overlaps.
        """
        try:
            col = self._store._client.get_collection(self._store.collection_name)
            where = None
            if self._source_filter:
                if len(self._source_filter) == 1:
                    where = {"source": self._source_filter[0]}
                else:
                    where = {"source": {"$in": self._source_filter}}
            
            if where:
                result = col.get(where=where, include=["metadatas", "documents"])
            else:
                result = col.get(include=["metadatas", "documents"])
                
            docs = []
            search_terms = []
            if condition:
                search_terms = [t.strip() for t in re.split(r'[\s\-]+', condition) if len(t.strip()) > 3]

            for meta, text in zip(result['metadatas'], result['documents']):
                heading = (meta.get("heading") or "").lower()
                text_lower = text.lower()
                match = False
                
                if search_terms and all(term in text_lower for term in search_terms):
                    match = True
                elif condition and condition in text_lower:
                    match = True
                
                if "critical illness" in heading or "critical illness" in text_lower:
                    match = True
                
                if any(k in heading or k in text_lower for k in ["cancer", "carcinoma", "melanoma", "eye surgery"]):
                    match = True
                    
                if match:
                    parent = meta.get("parent_text", text)
                    doc = Document(
                        page_content=parent,
                        metadata={
                            "source": meta.get("source"),
                            "page": meta.get("page"),
                            "heading": meta.get("heading", ""),
                            "clause": meta.get("clause", ""),
                            "parent_text": parent,
                        }
                    )
                    docs.append(doc)
            
            seen = set()
            unique = []
            for doc in docs:
                key = (doc.metadata.get("page"), doc.metadata.get("heading"))
                if key not in seen:
                    seen.add(key)
                    unique.append(doc)
            return unique[:5]
        except Exception as e:
            print(f"[chain] Error fetching forced chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # Retrieval with forced inclusion
    # ------------------------------------------------------------------

    def _retrieve_docs(self, query: str, debug: bool = False) -> List[Document]:
        q_lower = query.lower()
        is_targeted_query = any(term in q_lower for term in ["cancer", "carcinoma", "melanoma", "tumor", "eye", "surgery"])

        # 1. Regular hybrid search
        use_mmr = not is_targeted_query
        search_type = "mmr" if use_mmr else "similarity"

        hybrid_retriever = self._store.as_hybrid_retriever(
            k=HYBRID_FETCH_K,
            source_filter=self._source_filter,
            search_type=search_type,
        )
        raw_docs = hybrid_retriever.invoke(query)
        if not raw_docs:
            raw_docs = []

        # 2. Force-fetch structural conditional vectors if targeting specific clauses
        forced_docs = []
        if is_targeted_query:
            print("[chain] Target query identified – forcing structural retrieval path.")
            sys.stdout.flush()
            condition = self._extract_condition(query)
            forced_docs = self._fetch_forced_chunks(condition)
            sys.stdout.flush()

        # 3. Expand all raw docs to parent_text
        parent_map: Dict[str, Document] = {}
        for doc in raw_docs:
            parent = doc.metadata.get("parent_text", doc.page_content)
            if parent not in parent_map:
                parent_map[parent] = doc
        expanded = list(parent_map.values())

        if debug:
            print(f"[DEBUG] Expanded to {len(expanded)} unique parents.")
            sys.stdout.flush()

        # 4. Rerank expanded parents
        if expanded:
            cross_enc = self._get_cross_encoder()
            pairs = [(query, doc.page_content) for doc in expanded]
            scores = cross_enc.predict(pairs)
            sorted_pairs = sorted(zip(expanded, scores), key=lambda x: x[1], reverse=True)
            reranked = [doc for doc, _ in sorted_pairs]
        else:
            reranked = []

        # 5. Build final list: forced docs first, then reranked (excluding duplicates)
        final_docs = []
        forced_keys = set()
        for doc in forced_docs:
            key = (doc.metadata.get("page"), doc.metadata.get("heading"))
            forced_keys.add(key)
            final_docs.append(doc)

        for doc in reranked:
            key = (doc.metadata.get("page"), doc.metadata.get("heading"))
            if key not in forced_keys:
                final_docs.append(doc)

        # 6. Reorder
        if REORDER_ENABLED and final_docs:
            reorder = LongContextReorder()
            final_docs = reorder.transform_documents(final_docs)

        # 7. Truncate
        final_docs = final_docs[:self._k]

        if debug:
            print(f"[DEBUG] Final {len(final_docs)} docs.")
            sys.stdout.flush()
            for i, doc in enumerate(final_docs):
                heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
                page = doc.metadata.get("page", "?")
                print(f"  {i+1}: Page {page} | Heading: {heading}")
                print(f"      Content preview: {doc.page_content[:200]}...")
                sys.stdout.flush()
            print("="*60 + "\n")
            sys.stdout.flush()
        return final_docs

    # ------------------------------------------------------------------
    # Public debug method
    # ------------------------------------------------------------------

    def debug_retrieval(self, question: str) -> str:
        expanded = self._transform_query(question)
        print(f"\n[DEBUG] Transformed query: {expanded}")
        sys.stdout.flush()
        docs = self._retrieve_docs(expanded, debug=True)
        debug_lines = []
        debug_lines.append(f"Transformed query: {expanded}")
        debug_lines.append("="*60)
        debug_lines.append(f"Final {len(docs)} documents retrieved.")
        for i, doc in enumerate(docs):
            heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
            page = doc.metadata.get("page", "?")
            debug_lines.append(f"[{i+1}] Page: {page} | Heading: {heading}")
            debug_lines.append(f"    Content preview: {doc.page_content[:200]}...")
        debug_lines.append("="*60)
        return "\n".join(debug_lines)

    # ------------------------------------------------------------------
    # Formatting and history
    # ------------------------------------------------------------------

    def _format_docs(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            m = doc.metadata
            heading = m.get("heading", "") or m.get("clause", "")
            label = (
                f"[Passage {i} | Source: {m.get('source','?')} | "
                f"Page {m.get('page','?')}"
                + (f" | Section: {heading}" if heading else "")
                + "]"
            )
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n" + "="*60 + "\n\n".join(parts)

    def _trimmed_history(self) -> list:
        keep = self._window * 2
        return self._history[-keep:] if len(self._history) > keep else self._history[:]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        expanded = self._transform_query(question)
        docs = self._retrieve_docs(expanded, debug=False)
        context = self._format_docs(docs)

        messages = _PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )
        answer = self._parser.invoke(self._llm.invoke(messages))

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        sources = []
        seen = set()
        for doc in docs:
            m = doc.metadata
            key = f"{m.get('source')}::{m.get('page')}::{m.get('heading')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": m.get("source", ""),
                    "page": m.get("page", "?"),
                    "line": m.get("line", "?"),
                    "clause": m.get("heading", "") or m.get("clause", ""),
                    "snippet": doc.page_content[:180].replace("\n", " "),
                })

        return {"answer": answer, "sources": sources}

    def extract_attributes(self, source_filter: list[str] | None = None) -> dict:
        merged = {}
        sf = source_filter or self._source_filter

        for fields, query in _EXTRACT_GROUPS:
            results = self._store.retrieve(query=query, k=8, source_filter=sf)
            if not results:
                for f in fields:
                    merged[f] = None
                continue

            context_parts = []
            for r in results:
                parent = r.get("parent_text", r["text"])
                context_parts.append(f"[Page {r['page']} | {r.get('heading','')}]\n{parent}")
            context = "\n\n---\n\n".join(context_parts)

            prompt = _EXTRACT_SYSTEM.format(
                fields=json.dumps(fields),
                context=context,
            )
            raw = self._parser.invoke(
                self._llm.invoke([SystemMessage(content=prompt)])
            )
            raw = _strip_fences(raw)

            try:
                group_data = json.loads(raw)
                if isinstance(group_data, dict):
                    merged.update(group_data)
                else:
                    for f in fields:
                        merged[f] = None
            except json.JSONDecodeError:
                for f in fields:
                    merged.setdefault(f, None)
                merged.setdefault("_parse_errors", []).append(raw[:200])

        return merged

    def reset_memory(self) -> None:
        self._history.clear()

    def set_source_filter(self, sources: list[str] | None) -> None:
        self._source_filter = sources
        self.reset_memory()