"""
chain.py
--------------
PolicyChain — the main public interface.

Handles:
  - LLM initialisation (vLLM via OpenAI API)
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
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer

from constants import CHAT_PROMPT, FINAL_K, MODEL_NAME
from formatting import clean_output, extract_sources, format_docs
from retrieval import retrieve_docs, transform_query
from vector_store import PolicyVectorStore
from extract_attribute import run_extraction


class PolicyChain:

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        model: str = MODEL_NAME,
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

        print(f"[PolicyChain] Initialized with source_filter: {source_filter}")


        # Swapped ChatOllama for ChatOpenAI pointing to local vLLM server
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=384,
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="EMPTY",  # vLLM does not require a real API key
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        self._parser = StrOutputParser()
        self._tokenizer= AutoTokenizer.from_pretrained(model,trust_remote_code=True)




    def extract_attributes(self, source_filter=None):
        return run_extraction(store=self._store, llm= self._llm, parser=self._parser, source_filter=source_filter or self._source_filter)
    
    
    
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


    def _build_budgeted_context(self, docs, question: str):
        MAX_MODEL_CONTEXT = 4096
        MAX_OUTPUT_TOKENS = 384
        SAFETY_MARGIN = 150

        base_messages = CHAT_PROMPT.format_messages(
            context="",
            chat_history=self._trimmed_history(),
            question=question,
            active_policy_names="",  # placeholder for token counting only
        )
        base_tokens = sum(
            len(self._tokenizer.encode(m.content, add_special_tokens=False))
            for m in base_messages
        )
        available = MAX_MODEL_CONTEXT - MAX_OUTPUT_TOKENS - SAFETY_MARGIN - base_tokens

        # Group docs by policy, preserving their relevance order within each group
        from collections import defaultdict
        by_policy: dict[str, list] = defaultdict(list)
        order: list[str] = []
        for doc in docs:
            pid = doc.metadata.get("policy_id", "UNKNOWN")
            if pid not in by_policy:
                order.append(pid)
            by_policy[pid].append(doc)

        context_parts = []
        kept_docs = []
        used = 0
        pointers = {pid: 0 for pid in order}

        # Round-robin: take one doc from each policy in turn until budget runs out
        # or all policies are exhausted. This guarantees every policy gets a fair
        # shot at the budget instead of the first-listed policy consuming it all.
        progress = True
        while progress and used < available:
            progress = False
            for pid in order:
                idx = pointers[pid]
                if idx >= len(by_policy[pid]):
                    continue
                doc = by_policy[pid][idx]
                formatted = format_docs([doc])
                n_tokens = len(self._tokenizer.encode(formatted, add_special_tokens=False))

                if used + n_tokens > available:
                    pointers[pid] = len(by_policy[pid])  # stop pulling from this policy
                    continue

                context_parts.append(formatted)
                kept_docs.append(doc)
                used += n_tokens
                pointers[pid] = idx + 1
                progress = True

        print(
            f"[chain] Context budget: {used}/{available} tokens "
            f"({len(kept_docs)}/{len(docs)} docs across {len(order)} policies)"
        )

        return "\n\n".join(context_parts), kept_docs
    

    def _active_policy_names(self, docs) -> str:
        seen = []
        for doc in docs:
            pid = doc.metadata.get("policy_id", "UNKNOWN")
            if pid not in seen:
                seen.append(pid)
        if not seen and self._source_filter:
            seen = list(self._source_filter)
        return ", ".join(seen) if seen else "the selected policy"
        
    # ------------------------------------------------------------------
    # Public: chat
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        t0 = time.time()
        expanded = transform_query(question)
        t1 = time.time(); print(f"[chain] Query transformed : {t1-t0:.2f}s"); sys.stdout.flush()

        docs = self._retrieve(expanded)
        t2 = time.time(); print(f"[chain] Docs retrieved    : {t2-t1:.2f}s"); sys.stdout.flush()

        context, docs = self._build_budgeted_context(docs, question)
        active_names = self._active_policy_names(docs)          # <-- NEW

        messages = CHAT_PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
            active_policy_names=active_names,                    # <-- NEW
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
        expanded = transform_query(question)
        docs = self._retrieve(expanded)
        context, docs = self._build_budgeted_context(docs, question)
        active_names = self._active_policy_names(docs)          

        messages = CHAT_PROMPT.format_messages(
            context=context,
            chat_history=self._trimmed_history(),
            question=question,
            active_policy_names=active_names,                    
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
    # Public: memory / filter
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        self._history.clear()

    def set_source_filter(self, sources: list[str] | None) -> None:
        self._source_filter = sources
        self.reset_memory()