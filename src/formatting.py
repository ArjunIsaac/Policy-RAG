"""
formatting.py
-------------
Utilities for formatting retrieved docs into LLM context,
cleaning LLM output (stripping analysis tags), and
extracting citation sources from retrieved documents.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a numbered context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        m       = doc.metadata
        heading = m.get("heading", "") or m.get("clause", "")
        label   = (
            f"[Passage {i} | Source: {m.get('source', '?')} | "
            f"Page {m.get('page', '?')}"
            + (f" | Section: {heading}" if heading else "")
            + "]"
        )
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n" + "=" * 60 + "\n\n".join(parts)


def clean_output(raw: str) -> str:
    """
    Extract content inside <final_response> tags if present.
    Falls back to stripping <policy_analysis> tags and returning everything else.
    """
    m = re.search(r"<final_response>(.*?)</final_response>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    clean = re.sub(r"<policy_analysis>.*?</policy_analysis>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"</?final_response>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    return clean.strip()


def extract_sources(docs: List[Document]) -> list[dict]:
    """Build a deduplicated list of citation dicts from retrieved documents."""
    sources = []
    seen: set = set()
    for doc in docs:
        m   = doc.metadata
        key = f"{m.get('source')}::{m.get('page')}::{m.get('heading')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  m.get("source", ""),
                "page":    m.get("page", "?"),
                "line":    m.get("line", "?"),
                "clause":  m.get("heading", "") or m.get("clause", ""),
                "snippet": doc.page_content[:180].replace("\n", " "),
            })
    return sources