"""
chat_stream.py
--------------
PolicyChain — the main public interface.

Handles:
  - LLM initialisation (Mistral via Ollama)
  - Chat with memory (ask / ask_stream)
  - Debug retrieval
  - Attribute extraction (delegates to attribute_extract.py)
"""

from __future__ import annotations

import sys
import time
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from attribute_extract import run_extraction
from constants import CHAT_PROMPT, FINAL_K
from formatting import clean_output, extract_sources, format_docs
from retrieval import retrieve_docs, transform_query
from vector_store import PolicyVectorStore


class PolicyChain:
    """
    Main entry point for the Policy Interrogator RAG system.

    Chat pipeline  : ask() / ask_stream()
    Attribute panel: extract_attributes()
    Debug tool     : debug_retrieval()
    """

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        model: str = "mistral",
        temperature: float = 0.0,
        k_docs: int = FINAL_K,
        memory_window: int = 4,
        source_filter: list[str] | None = None,
    ) -> None:
        self._store         = vector_store
        self._k             = k_docs
        self._source_filter = source_filter
        self._window        = memory_window
        self._history: list[HumanMessage | AIMessage] = []

        self._llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=2048,
            num_ctx=8192,
            extra_body={"think": False},
        )
        self._parser = StrOutputParser()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trimmed_history(self) -> list:
        keep = self._window * 2
        return self._history[-keep:] if len(self._history) > keep else self._history[:]

    def _retrieve(self, query: str, debug: bool = False):
        return retrieve_docs(
            store=self._store,
            query=query,
            source_filter=self._source_filter,
            k=self._k,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Public: chat
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """Synchronous ask — returns {answer, sources}."""
        t0 = time.time()

        expanded = transform_query(question)
        t1 = time.time(); print(f"[chain] Query transformed : {t1-t0:.2f}s"); sys.stdout.flush()

        docs     = self._retrieve(expanded)
        t2 = time.time(); print(f"[chain] Docs retrieved    : {t2-t1:.2f}s"); sys.stdout.flush()

        context  = format_docs(docs)
        messages = CHAT_PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )
        t3 = time.time(); print(f"[chain] Context built     : {t3-t2:.2f}s"); sys.stdout.flush()

        raw_answer = self._parser.invoke(self._llm.invoke(messages))
        answer     = clean_output(raw_answer)
        t4 = time.time(); print(f"[chain] LLM generation    : {t4-t3:.2f}s"); sys.stdout.flush()
        print(f"[chain] Total             : {t4-t0:.2f}s"); sys.stdout.flush()

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        return {"answer": answer, "sources": extract_sources(docs)}

    def ask_stream(self, question: str):
        """Streaming ask — yields {type: 'text'|'sources', ...} dicts."""
        expanded = transform_query(question)
        docs     = self._retrieve(expanded)
        context  = format_docs(docs)
        messages = CHAT_PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
        )

        full_raw        = ""
        in_final        = False
        buffer          = ""

        for chunk in self._llm.stream(messages):
            content   = chunk.content
            full_raw += content

            if not in_final:
                buffer += content
                if "<final_response>" in buffer:
                    in_final = True
                    after = buffer.split("<final_response>", 1)[1]
                    clean = after.replace("</final_response>", "")
                    if clean:
                        yield {"type": "text", "content": clean}
                    buffer = ""
            else:
                if "</final_response>" in content:
                    clean = content.split("</final_response>")[0]
                    if clean:
                        yield {"type": "text", "content": clean}
                else:
                    yield {"type": "text", "content": content}

        # Fallback: model skipped structural tags entirely
        if not in_final and full_raw:
            yield {"type": "text", "content": clean_output(full_raw)}

        answer = clean_output(full_raw)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        yield {"type": "sources", "sources": extract_sources(docs)}

    # ------------------------------------------------------------------
    # Public: debug
    # ------------------------------------------------------------------

    def debug_retrieval(self, question: str) -> str:
        expanded = transform_query(question)
        print(f"\n[DEBUG] Transformed query: {expanded}")
        sys.stdout.flush()
        docs = self._retrieve(expanded, debug=True)

        lines = [
            f"Transformed query: {expanded}",
            "=" * 60,
            f"Final {len(docs)} documents retrieved.",
        ]
        for i, doc in enumerate(docs):
            heading = doc.metadata.get("heading", "") or doc.metadata.get("clause", "")
            lines.append(f"[{i+1}] Page: {doc.metadata.get('page','?')} | Heading: {heading}")
            lines.append(f"    Content preview: {doc.page_content[:200]}...")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public: attribute extraction
    # ------------------------------------------------------------------

    def extract_attributes(self, source_filter: list[str] | None = None) -> dict:
        """
        Extract partner-relevant policy attributes.
        Fixed fields  : regex, instant, zero LLM calls.
        Dynamic fields: 1 LLM call (~30s).
        Completely independent of the RAG chat pipeline.
        """
        sf = source_filter or self._source_filter
        return run_extraction(self._store, self._llm, self._parser, sf)

    # ------------------------------------------------------------------
    # Public: memory / filter
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        self._history.clear()

    def set_source_filter(self, sources: list[str] | None) -> None:
        self._source_filter = sources
        self.reset_memory()