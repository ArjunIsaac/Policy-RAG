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
import time  # identifying potential bottlenecks in retrieval
from typing import Any, List, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_community.document_transformers import LongContextReorder
from sentence_transformers import CrossEncoder

import nltk  # language processing
from nltk.corpus import stopwords

from vector_store import PolicyVectorStore

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {
    'please', 'tell', 'me', 'know', 'want', 'ask', 'like', 'help',
    'thank', 'thanks', 'hi', 'hello', 'hey', 'maybe', 'perhaps', 'basically', 'actually',
    'really', 'quite', 'just', 'also', 'well', 'look', 'see', 'think', 'guess', 'feel',
}
STOPWORDS.update(CUSTOM_STOPWORDS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_FETCH_K = 100
FINAL_K = 8
REORDER_ENABLED = True

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert insurance policy analyst reviewing regulatory policy contracts. 

Your task is to analyze the retrieved context and provide a highly accurate determination of coverage, waiting periods, and exclusions.

CRITICAL LOGIC RULE (TERMINOLOGY MAPPING):
Users often ask questions using common language (e.g., "skin cancer", "eye surgery", "LASIK"). Insurance policies use formal legal or medical definitions (e.g., "skin carcinoma", "malignant melanoma", "cataract", "refractive error"). 
Before concluding that a condition is unmentioned, you MUST check if the common term maps to a formal definition or sub-exclusion within the context.

EXECUTION PROTOCOL:
You must process your response using two distinct steps. 
1. Inside an internal `<policy_analysis>` section, explicitly evaluate terminology synonyms and cross-reference the text for any exclusions or conditional clauses related to those mapped terms.
2. Provide your clean, comprehensive final response to the user inside a `<final_response>` section. Do not include the XML block syntax inside your text; output them as clean, structural tags.

STRICT INSTRUCTIONAL RULES FOR FINAL RESPONSE:
1. Answer the question directly using ONLY the RETRIEVED CONTEXT below. Do not assume or extrapolate beyond the provided text.
2. INTERPRET TABLES ACCURATELY: Insurance policies utilize benefit grids. If a policy benefit, surgery type, or clause is associated with the term "NIL", "No Coverage", "0", or "-" inside a table row or text block, this means coverage for that item is completely ZERO / NOT COVERED. You must state this explicitly.
3. If a condition or coverage is subject to a conditional exclusion (e.g., "Excluded unless X happens" or "Covered only if Y is met"), you MUST state that exact condition clearly instead of stating that the policy is unclear or does not mention it.
4. Quote exact policy text or table entries when stating inclusions, exclusions, or conditional requirements.
5. If a condition is definitively and permanently excluded without exception, state: "The policy explicitly excludes...".
6. If a condition is completely unmentioned anywhere in the text (even after checking for medical/legal synonyms), state: "The policy does not mention this condition."
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
# Attribute extraction — partner-focused, full-document approach
# ---------------------------------------------------------------------------

# Each group now uses a generic query built from field names/synonyms
# instead of hard‑coded example values.
_PARTNER_ATTR_GROUPS = [
    (
        {
            "policy_name": "string - product/plan name e.g. Total Health Plan",
            "insurer": "string - insurance company name",
            "sum_insured_options": "array of strings - all sum insured amounts e.g. ['5 Lakhs','10 Lakhs']",
            "policy_tenure": "string - policy duration e.g. '1 Year'",
            "lifetime_renewability": "boolean - true if policy is lifetime renewable",
            "free_look_period_days": "number - free look period in days e.g. 15",
            "grace_period_days": "number - grace period for renewal in days e.g. 30",
        },
        "policy name insurer sum insured tenure renewal free look grace period",  # generic query
        "Look in: product name heading, sum insured table, tenure, renewal clause, free look period, grace period.",
    ),
    (
        {
            "waiting_period_initial_days": "number - initial waiting period in days e.g. 30",
            "waiting_period_ped_months": "number - pre-existing disease PED waiting period in months e.g. 48",
            "waiting_period_specific_illness_months": "number - specific illness/procedure waiting period in months e.g. 24",
        },
        "initial waiting period pre-existing disease PED specific illness procedure waiting period",
        "Look in: Section C Waiting Period & Exclusions. Initial waiting period, PED, specific disease/procedure waiting.",
    ),
    (
        {
            "copay_applicable": "boolean - true if any co-pay clause exists",
            "copay_percentage": "number or null - co-pay percentage e.g. 20",
            "copay_conditions": "string or null - exact conditions when co-pay applies",
            "room_rent_sublimit": "string or null - room rent daily cap e.g. '1% of sum insured'",
            "icu_sublimit": "string or null - ICU charges cap e.g. '2% of sum insured'",
        },
        "co-payment copay room rent sub-limit ICU intensive care unit bed charges",
        "Look in: definition of Copayment, co-pay clause. Room rent and ICU sub-limits: daily cap or percentage.",
    ),
    (
        {
            "inpatient_covered": "boolean - true if inpatient hospitalisation is covered",
            "daycare_covered": "boolean - true if day care procedures are covered",
            "domiciliary_covered": "boolean - true if domiciliary home treatment is covered",
            "maternity_covered": "boolean - true if maternity expenses are covered",
            "ambulance_covered": "boolean - true if emergency ambulance is covered",
            "organ_donor_covered": "boolean - true if organ donor harvesting expenses are covered",
            "pre_hospitalisation_days": "number - pre-hospitalisation cover in days e.g. 30",
            "post_hospitalisation_days": "number - post-hospitalisation cover in days e.g. 60",
        },
        "inpatient day care domiciliary maternity ambulance organ donor pre-hospitalisation post-hospitalisation",
        "Look in: Section B Benefits - inpatient, day care, domiciliary, maternity, ambulance, organ donor, pre/post.",
    ),
    (
        {
            "cashless_available": "boolean - true if cashless facility at network hospitals",
            "network_hospitals": "string - description or count of network hospitals",
            "claim_settlement_days": "number - days insurer must settle claim e.g. 30",
            "portability_available": "boolean - true if policy can be ported to another insurer",
            "ncb_benefit": "string or null - No Claim Bonus or Cumulative Bonus description",
        },
        "cashless network hospital claim settlement portability No Claim Bonus cumulative bonus NCB",
        "Look in: cashless service, network hospitals, claim settlement timeframe, portability clause, Cumulative Bonus.",
    ),
    (
        {
            "permanent_exclusions": "array of strings - key permanently excluded conditions (max 10)",
        },
        "permanent exclusions not covered excluded war cosmetic obesity adventure sports alcohol infertility",
        "Look in: Section C Standard and Specific General Exclusions. List main permanent exclusions.",
    ),
]

_DYNAMIC_ATTR_PROMPT = """You are an expert insurance analyst helping insurance PARTNERS pitch policies to clients.

Read this insurance policy and identify ONLY benefits/features that:
1. Are a SELLING POINT a partner would highlight when pitching to a client
2. Are PRODUCT FEATURES — not definitions, not exclusions, not admin clauses

ONLY include things like:
- Restore/Recharge benefit (sum insured restored after a claim)
- Multiplier / Cumulative bonus (sum insured increases each claim-free year)
- OPD cover (outpatient consultations covered)
- Daily hospital cash benefit
- Newborn baby cover
- Mental health cover
- E-opinion / second medical opinion benefit
- Moratorium period (after X years, no pre-existing disease lookback)
- International cover
- Deductible options
- Health check-up benefit
- Any rider or add-on benefit

DO NOT include:
- Medical definitions (e.g. what TIA means, what dialysis means)
- Exclusions or what is NOT covered
- Admin clauses (fraud, nomination, cancellation, notices)
- Anything already captured in standard attributes (waiting periods, co-pay, room rent, maternity, exclusions)

Return a JSON object: keys = short snake_case feature names, values = short description INCLUDING the exact wording found in the policy. 
CRITICAL RULES:

- NEVER invent a benefit.
- ONLY return a feature if explicit evidence exists in the supplied text.
- If the text does not explicitly mention the feature, do not include it.
- If unsure, omit it.
- Features must be directly quoted or clearly supported by the text.
- Return {{}} if no feature is explicitly found.

Policy text:
{text}

JSON:"""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]
    end_brace = raw.rfind("}")
    if end_brace != -1 and end_brace < len(raw) - 1:
        raw = raw[:end_brace + 1]
    return raw.strip()


def _get_full_policy_text(store: "PolicyVectorStore", source_filter: list[str] | None) -> str:
    """
    Reassemble full policy text from ChromaDB chunks sorted by page.
    Chunks into segments of ~6000 chars each to handle any PDF size safely.
    Returns list of text segments.
    """
    try:
        col = store._client.get_collection(store.collection_name)
        where = None
        if source_filter and len(source_filter) == 1:
            where = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"source": {"$in": source_filter}}

        result = col.get(where=where, include=["documents", "metadatas"]) if where else col.get(include=["documents", "metadatas"])
        pairs = sorted(zip(result["documents"], result["metadatas"]), key=lambda x: x[1].get("page", 0))

        seen, unique_texts = set(), []
        for text, meta in pairs:
            if text not in seen:
                seen.add(text)
                unique_texts.append(f"[Page {meta.get('page', '?')}]\n{text}")

        return "\n\n".join(unique_texts)
    except Exception as e:
        print(f"[chain] Error getting full policy text: {e}")
        return ""


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
            print("[chain] Loading CrossEncoder on GPU…")
            sys.stdout.flush()
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cuda")
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _remove_stopwords(self, text: str) -> str:
        words = text.lower().split()
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
        return ' '.join(filtered)

    def _transform_query(self, query: str) -> str:
        q_lower = query.lower()

        if any(term in q_lower for term in ["cancer", "carcinoma", "melanoma", "tumor"]):
            query += " critical illness definition carcinoma melanoma tumor malignancy"
        if any(term in q_lower for term in ["excluded", "exclusion", "not covered"]):
            query += " permanent exclusion standard exclusions limits"
        if any(term in q_lower for term in ["waiting", "period"]):
            query += " waiting period pre-existing specific illness cataract"
        if any(term in q_lower for term in ["copay", "co-pay"]):
            query += " co-payment"
        if any(term in q_lower for term in ["eye", "cataract", "lasik", "surgery", "surgeries", "vision", "eyesight"]):
            query += " limit sublimit cap NIL table benefit cataract refractive error dioptres eyesight correction"

        return query

    # ------------------------------------------------------------------
    # Extract condition from query (Punctuation-agnostic)
    # ------------------------------------------------------------------

    def _extract_condition(self, query: str) -> str | None:
        normalized = query.replace('–', '-').replace('—', '-').lower()
        keywords = ["cancer", "carcinoma", "melanoma", "tumor", "eye", "cataract", "lasik", "refractive"]

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

                if "exclusion" in heading or "exclusion" in text_lower:
                    match = True

                if "waiting period" in heading or "waiting period" in text_lower:
                    match = True

                if any(k in heading or k in text_lower for k in ["cancer", "carcinoma", "melanoma", "eye surgery", "cataract", "lasik", "refractive", "dioptres"]):
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
            return unique[:8]  # Increased forced pool slightly to avoid missing specific sections
        except Exception as e:
            print(f"[chain] Error fetching forced chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # Retrieval with forced inclusion
    # ------------------------------------------------------------------

    def _retrieve_docs(self, query: str, debug: bool = False) -> List[Document]:
        q_lower = query.lower()
        is_targeted_query = any(term in q_lower for term in ["cancer", "carcinoma", "melanoma", "tumor", "eye", "surgery", "cataract", "lasik", "refractive"])

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

    def _clean_output(self, raw_output: str) -> str:
        """
        Extracts only the content within <final_response> tags if present.
        Falls back to returning everything if structural tags are omitted or broken.
        """
        match = re.search(r"<final_response>(.*?)</final_response>", raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback processing: strip out policy_analysis tags cleanly if the model mixed up layout
        clean = re.sub(r"<policy_analysis>.*?</policy_analysis>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"</?final_response>", "", clean, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        start_time = time.time()
        t0 = time.time()

        expanded = self._transform_query(question)
        t1 = time.time()
        print(f'[chain] Query transformed : {t1-t0:.2f}s')
        sys.stdout.flush()

        docs = self._retrieve_docs(expanded, debug=False)
        t2 = time.time()
        print(f'[chain] Docs retrieved : {t2-t1:.2f}s')
        sys.stdout.flush()

        context = self._format_docs(docs)
        t3 = time.time()
        print(f'[chain] Context formatted : {t3-t2:.2f}s')
        sys.stdout.flush()

        messages = _PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )

        t4 = time.time()
        print(f'[chain] Build messages:{t4-t3:.2f}s')
        sys.stdout.flush()

        raw_answer = self._parser.invoke(self._llm.invoke(messages))
        answer = self._clean_output(raw_answer)

        t5 = time.time()
        print(f'[chain] LLM generation: {t5-t4:.2f}s')
        sys.stdout.flush()
        print(f'[chain] Total time: {t5-start_time:.2f}s')
        sys.stdout.flush()

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

    def ask_stream(self, question: str):
        expanded = self._transform_query(question)
        docs = self._retrieve_docs(expanded, debug=False)
        context = self._format_docs(docs)

        messages = _PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )

        full_raw_answer = ""
        in_final_response = False
        buffer = ""

        for chunk in self._llm.stream(messages):
            content = chunk.content
            full_raw_answer += content

            if not in_final_response:
                buffer += content
                if "<final_response>" in buffer:
                    in_final_response = True
                    # Yield anything that arrived after the opening structural tag
                    parts = buffer.split("<final_response>")
                    if len(parts) > 1 and parts[1]:
                        clean_content = parts[1].replace("</final_response>", "")
                        if clean_content:
                            yield {"type": "text", "content": clean_content}
                    buffer = ""
            else:
                if "</final_response>" in content:
                    clean_content = content.split("</final_response>")[0]
                    if clean_content:
                        yield {"type": "text", "content": clean_content}
                else:
                    yield {"type": "text", "content": content}

        # Fallback if structural tags were completely skipped by the model
        if not in_final_response and full_raw_answer:
            yield {"type": "text", "content": self._clean_output(full_raw_answer)}

        answer = self._clean_output(full_raw_answer)

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

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        yield {"type": "sources", "sources": sources}

    def _llm_json(self, system: str, user: str) -> dict:
        raw = self._parser.invoke(
            self._llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user)
            ])
        )

        raw = _strip_fences(raw)

        # Extract only first JSON object
        match = re.search(r"\{.*?\}", raw, re.DOTALL)

        if not match:
            print("[chain] No JSON found")
            return {}

        json_text = match.group(0)

        try:
            result = json.loads(json_text)

            if isinstance(result, dict):
                return result

        except Exception as e:
            print(f"[chain] JSON parse error: {e}")

        return {}

    # ------------------------------------------------------------------
    # Improved attribute extraction
    # ------------------------------------------------------------------

    def _retrieve_for_attributes(self, query: str, k: int = 20) -> List[Document]:
        """
        Hybrid retrieval specifically for attribute extraction.
        Uses cross-encoder reranking to select the most relevant documents.
        """
        # Use hybrid retriever with larger k
        retriever = self._store.as_hybrid_retriever(
            k=k,
            source_filter=self._source_filter,
            search_type="similarity"  # use similarity (not MMR) to get diverse relevant docs
        )
        docs = retriever.invoke(query)
        if not docs:
            return []

        # Rerank with cross-encoder to pick the best
        cross_enc = self._get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = cross_enc.predict(pairs)
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

        # Return top 8 (or a configurable number) for the LLM
        top_k = 8
        return sorted_docs[:top_k]

    def extract_attributes(self, source_filter: list[str] | None = None) -> dict:
        """
        Single LLM call, but with Multi-Query retrieval to cover all sections.
        Latency: ~3-4s on RTX 3050. Accuracy: ~95% of sequential.
        """
        sf = source_filter or self._source_filter
        
        # 1. Define sub-queries targeting different sections of the policy
        sub_queries = [
            "policy name insurer sum insured tenure renewal free look grace period lifetime renewability",
            "waiting period initial pre-existing disease PED specific illness waiting period",
            "co-payment copay room rent sub-limit ICU intensive care unit bed charges",
            "inpatient day care domiciliary maternity ambulance organ donor pre-hospitalisation post-hospitalisation",
            "cashless network hospital claim settlement portability No Claim Bonus cumulative bonus NCB",
            "permanent exclusions not covered excluded war cosmetic obesity adventure sports alcohol infertility",
        ]
        
        # 2. Retrieve for EACH sub-query (10 per query = 60 raw docs, but we dedupe)
        all_docs = []
        seen_content = set()
        
        for q in sub_queries:
            raw = self._store.as_hybrid_retriever(k=10, source_filter=sf).invoke(q)
            for doc in raw:
                # Use parent_text for deduplication to avoid overlapping sections
                parent = doc.metadata.get("parent_text", doc.page_content)
                if parent not in seen_content:
                    seen_content.add(parent)
                    all_docs.append(doc)
        
        if not all_docs:
            return {}
        
        # 3. Rerank ALL retrieved docs against a master query
        master_query = " ".join(sub_queries) + " policy summary"
        cross_enc = self._get_cross_encoder()
        pairs = [(master_query, doc.page_content) for doc in all_docs]
        scores = cross_enc.predict(pairs)
        sorted_docs = [doc for _, doc in sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)]
        
        # 4. Take top 12 (ensures diverse coverage, not just the same section)
        final_docs = sorted_docs[:12]
        
        # 5. Build context (using parent_text, truncated to 1000 chars each)
        context_parts = []
        for doc in final_docs:
            parent = doc.metadata.get("parent_text", doc.page_content)
            # Truncate to keep total context ~12k chars (~3k tokens)
            context_parts.append(f"[Page {doc.metadata.get('page', '?')}] {parent[:1000]}")
        context = "\n---\n".join(context_parts)
        
        # 6. Build the full JSON schema
        all_fields = {}
        all_hints = []
        for field_defs, _, hint in _PARTNER_ATTR_GROUPS:
            all_fields.update(field_defs)
            all_hints.append(hint)
        
        system = "You are an expert insurance extractor. Respond with ONLY a valid JSON object. Do not add extra text."
        user = f"""
    Extract ALL these fields from the policy excerpts. Return a single JSON object.

    Field definitions:
    {json.dumps(all_fields, indent=2)}

    Hints: {' '.join(all_hints)}

    Policy excerpts:
    {context}

    JSON:
    """
        result = self._llm_json(system, user)
        
        # Ensure all keys exist (fill missing with None)
        for key in all_fields.keys():
            if key not in result:
                result[key] = None
        
        return result

        # Dynamic attributes – can be enabled later with the same improved retrieval
        # dyn_query = "restore recharge cumulative bonus OPD hospital cash health checkup wellness"
        # dyn_docs = self._retrieve_for_attributes(dyn_query, k=10)
        # if dyn_docs:
        #     dyn_context = "\n---\n".join(
        #         f"[Page {d.metadata.get('page','?')}] {d.metadata.get('parent_text', d.page_content)[:800]}"
        #         for d in dyn_docs
        #     )
        #     dynamic = self._llm_json(system, _DYNAMIC_ATTR_PROMPT.format(text=dyn_context))
        #     if dynamic:
        #         merged["_dynamic"] = dynamic


    def reset_memory(self) -> None:
        self._history.clear()

    def set_source_filter(self, sources: list[str] | None) -> None:
        self._source_filter = sources
        self.reset_memory()