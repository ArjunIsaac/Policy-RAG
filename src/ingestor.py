"""
ingestor.py
-----------
Parses insurance policy PDFs using pdfplumber.
Extracts text and tables, chunks with sliding window, and records
page number + approximate line/clause number for citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pdfplumber


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Insurance keyword detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    r"waiting period[s]?",
    r"co[\-\s]?pay[ment]*",
    r"sub[\-\s]?limit[s]?",
    r"exclusion[s]?",
    r"pre[\-\s]?existing",
    r"deductible[s]?",
    r"premium[s]?",
    r"benefit[s]?",
    r"coverage|covered",
    r"network hospital[s]?",
    r"claim[s]? procedure",
    r"grace period",
    r"renewal",
    r"sum insured",
    r"room rent",
    r"icu charges?",
    r"day care",
    r"maternity",
    r"ambulance",
]

_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE)

# Matches things like "1.", "1.1", "a)", "(i)", "Clause 3", "Section 4"
_CLAUSE_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:clause|section|article|para(?:graph)?|point)?\s*"
    r"(?:\d+(?:\.\d+)*[.):]|[a-zA-Z][.):]|\([ivxlIVXL\d]+\))",
    re.IGNORECASE,
)


def _detect_tags(text: str) -> list[str]:
    return list({m.group(0).strip().lower() for m in _SECTION_RE.finditer(text)})


def _detect_clause(text: str) -> str:
    """Return the first clause/section marker found in the chunk, or ''."""
    m = _CLAUSE_RE.search(text)
    return m.group(0).strip() if m else ""


def _table_to_text(table: list[list[str | None]]) -> str:
    if not table:
        return ""
    lines: list[str] = []
    for row in table:
        clean = [str(cell).strip() if cell is not None else "" for cell in row]
        lines.append(" | ".join(clean))
    if len(lines) > 1:
        lines.insert(1, "-" * max(len(l) for l in lines))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sliding-window chunker  (word-level, preserves line numbers)
# ---------------------------------------------------------------------------

def _sliding_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> Generator[tuple[str, int], None, None]:
    """
    Yield (chunk_text, start_line) pairs.
    start_line is the 1-based line number where this chunk begins.
    """
    lines = text.split("\n")
    words_with_lines: list[tuple[str, int]] = []
    for lineno, line in enumerate(lines, start=1):
        for word in line.split():
            words_with_lines.append((word, lineno))

    if not words_with_lines:
        return

    start = 0
    while start < len(words_with_lines):
        end = min(start + chunk_size, len(words_with_lines))
        chunk_words = words_with_lines[start:end]
        text_chunk = " ".join(w for w, _ in chunk_words)
        start_line = chunk_words[0][1]
        yield text_chunk, start_line
        if end >= len(words_with_lines):
            break
        start += chunk_size - overlap


# ---------------------------------------------------------------------------
# Public ingestor
# ---------------------------------------------------------------------------

class PDFIngestor:
    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _process_page(
        self, page: pdfplumber.page.Page, source: str, page_num: int
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        table_header_texts: list[str] = []

        # 1. Tables
        try:
            for table in page.extract_tables() or []:
                table_text = _table_to_text(table)
                if not table_text.strip():
                    continue
                header = table_text.split("\n")[0]
                table_header_texts.append(header)
                tags = _detect_tags(table_text)
                clause = _detect_clause(table_text)
                chunks.append(Chunk(
                    text=table_text,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "line": 1,
                        "clause": clause,
                        "type": "table",
                        "tags": tags,
                    },
                ))
        except Exception:
            pass

        # 2. Prose
        try:
            prose = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        except Exception:
            prose = ""

        for hdr in table_header_texts:
            prose = prose.replace(hdr, "", 1)
        prose = prose.strip()

        if not prose:
            return chunks

        for chunk_text, start_line in _sliding_chunks(prose, self.chunk_size, self.overlap):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            tags = _detect_tags(chunk_text)
            clause = _detect_clause(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "source": source,
                    "page": page_num,
                    "line": start_line,
                    "clause": clause,
                    "type": "text",
                    "tags": tags,
                },
            ))

        return chunks

    def ingest(self, pdf_path: str | Path) -> list[Chunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source = pdf_path.name
        all_chunks: list[Chunk] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            print(f"[ingestor] '{source}' — {total} pages")
            for i, page in enumerate(pdf.pages, start=1):
                all_chunks.extend(self._process_page(page, source, i))
                if i % 10 == 0 or i == total:
                    print(f"[ingestor]   {i}/{total} pages, {len(all_chunks)} chunks")

        print(f"[ingestor] Done. Total chunks: {len(all_chunks)}")
        return all_chunks