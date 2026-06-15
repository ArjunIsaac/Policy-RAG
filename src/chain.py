"""
chain.py
--------
LCEL-based RAG chain. Replaces the deprecated ConversationalRetrievalChain.

LLM    : Mistral via Ollama
Memory : manual list of HumanMessage / AIMessage (last N turns)
Prompt : strict compliance-oriented system prompt
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert insurance policy analyst helping financial \
advisors understand complex policy documents quickly and accurately.

STRICT RULES — follow these without exception:
1. Answer ONLY from the retrieved context passages provided. Never invent figures.
2. If context is insufficient, reply exactly:
   "I could not find sufficient information in the provided policy document(s)."
3. For EVERY specific fact (waiting period, co-pay %, sub-limit, exclusion),
   include a citation in the format: (Source: <filename>, Page <N>, Line ~<L>)
   If a clause marker is available, also add it: (Clause <X>)
4. Present extracted values in labelled bullet format:
   • Waiting Period: 30 days  (Source: hdfc_policy.pdf, Page 12, Line ~45)
5. If asked to compare policies, use a side-by-side table with columns per policy.
6. Keep answers concise. Use bullet points for lists of clauses or exclusions.
7. If the question is ambiguous, ask ONE clarifying question before answering.

RETRIEVED CONTEXT:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Attribute extraction prompt (one-shot, no memory needed)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You are an insurance policy data extractor.
From the context below, extract the following attributes.
Return ONLY valid JSON — no markdown fences, no preamble.

Attributes to extract:
- policy_name
- insurer
- sum_insured  (list all options, e.g. ["5L","10L"])
- waiting_period_initial  (days for initial waiting period)
- waiting_period_ped  (days for pre-existing disease)
- waiting_period_specific  (days for specific illnesses)
- copay_percentage  (number or null)
- copay_conditions  (string describing when co-pay applies, or null)
- room_rent_sublimit  (per day limit or "No limit")
- icu_sublimit
- maternity_covered  (true/false/null)
- daycare_procedures  (number or "All" or null)
- exclusions_permanent  (list of strings, top 5 only)
- grace_period_days
- renewal_type  (e.g. "Lifelong renewable")
- ncb_benefit  (No Claim Bonus description or null)
- network_hospitals  (number or null)

CONTEXT:
{context}

JSON:"""

# ---------------------------------------------------------------------------
# PolicyChain
# ---------------------------------------------------------------------------

class PolicyChain:
    """
    Stateful RAG chain with manual conversation memory.

    Parameters
    ----------
    vector_store   : initialised PolicyVectorStore
    model          : Ollama model name
    temperature    : keep low for factual answers
    k_docs         : context chunks per query
    memory_window  : how many past Q&A turns to keep
    source_filter  : list of PDF filenames to restrict retrieval to
    """

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        model: str = "mistral",
        temperature: float = 0.05,
        k_docs: int = 6,
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
            num_predict=1024,
        )
        self._parser = StrOutputParser()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_docs(self, docs) -> str:
        """Turn retrieved documents into a context string with citations."""
        parts = []
        for doc in docs:
            m = doc.metadata
            citation = (
                f"[Source: {m.get('source','?')}, "
                f"Page {m.get('page','?')}, "
                f"Line ~{m.get('line','?')}"
            )
            clause = m.get("clause", "")
            if clause:
                citation += f", Clause: {clause}"
            citation += "]"
            parts.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _trimmed_history(self) -> list:
        """Return last N*2 messages (N user + N assistant turns)."""
        keep = self._window * 2
        return self._history[-keep:] if len(self._history) > keep else self._history[:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Send a question to the chain.

        Returns
        -------
        dict:
            answer  : str
            sources : list[dict]  — {source, page, line, clause, snippet}
        """
        retriever = self._store.as_retriever(
            k=self._k, source_filter=self._source_filter
        )
        docs = retriever.invoke(question)
        context = self._format_docs(docs)

        messages = _PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )

        answer = self._parser.invoke(self._llm.invoke(messages))

        # Update memory
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        # Build source list (deduplicated)
        sources = []
        seen: set[str] = set()
        for doc in docs:
            m = doc.metadata
            key = f"{m.get('source')}::{m.get('page')}::{m.get('line')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": m.get("source", ""),
                    "page": m.get("page", "?"),
                    "line": m.get("line", "?"),
                    "clause": m.get("clause", ""),
                    "snippet": doc.page_content[:180].replace("\n", " "),
                })

        return {"answer": answer, "sources": sources}

    def extract_attributes(self, source_filter: list[str] | None = None) -> dict:
        """
        Run a one-shot extraction of key policy attributes from the store.
        Returns a dict (parsed from LLM JSON output).
        """
        import json as _json

        # Pull a broad set of chunks for extraction
        retriever = self._store.as_retriever(k=20, source_filter=source_filter)
        docs = retriever.invoke(
            "waiting period co-pay sub-limit exclusion sum insured premium renewal"
        )
        context = self._format_docs(docs)

        prompt_text = EXTRACT_PROMPT.format(context=context)
        raw = self._parser.invoke(
            self._llm.invoke([SystemMessage(content=prompt_text)])
        )

        # Strip accidental markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            return {"_raw": raw, "_error": "JSON parse failed"}

    def reset_memory(self) -> None:
        self._history.clear()

    def set_source_filter(self, sources: list[str] | None) -> None:
        self._source_filter = sources
        self.reset_memory()