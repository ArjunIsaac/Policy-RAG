

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
    """A single piece of content extracted from a PDF."""
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    r"(waiting period[s]?)",
    r"(co[\-\s]?pay[ment]*)",
    r"(sub[\-\s]?limit[s]?)",
    r"(exclusion[s]?)",
    r"(pre[\-\s]?existing)",
    r"(deductible[s]?)",
    r"(premium[s]?)",
    r"(benefit[s]?)",
    r"(coverage|covered)",
    r"(network hospital[s]?)",
    r"(claim[s]? procedure)",
    r"(grace period)",
    r"(renewal)",
    r"(sum insured)",
]

_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE)


def _detect_section_tags(text: str) -> list[str]:
    """Return a deduplicated list of insurance-specific tags found in text."""
    return list({m.group(0).lower() for m in _SECTION_RE.finditer(text)})


def _table_to_text(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table (list of rows) into a Markdown-style string."""
    if not table:
        return ""
    lines: list[str] = []
    for row in table:
        clean = [str(cell).strip() if cell is not None else "" for cell in row]
        lines.append(" | ".join(clean))
    # Insert a separator after the header row
    if len(lines) > 1:
        lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

def _sliding_window_chunks(
    text: str,
    chunk_size: int = 600,
    overlap: int = 120,
) -> Generator[str, None, None]:
    """
    Yield overlapping text windows.
    Splitting on whitespace boundaries keeps words intact.
    """
    words = text.split()
    if not words:
        return
    start = 0
    while start < len(words):
        end = start + chunk_size
        yield " ".join(words[start:end])
        if end >= len(words):
            break
        start += chunk_size - overlap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PDFIngestor:
    """
    Reads a PDF with pdfplumber and returns a list of Chunk objects,
    each annotated with page number, source filename, and detected tags.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 120,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_page(
        self,
        page: pdfplumber.page.Page,
        source: str,
        page_num: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # 1. Extract tables first so their text doesn't muddy the prose
        table_texts: list[str] = []
        try:
            tables = page.extract_tables()
            for table in tables:
                table_text = _table_to_text(table)
                if table_text.strip():
                    table_texts.append(table_text)
                    tags = _detect_section_tags(table_text)
                    chunks.append(
                        Chunk(
                            text=table_text,
                            metadata={
                                "source": source,
                                "page": page_num,
                                "type": "table",
                                "tags": tags,
                            },
                        )
                    )
        except Exception:
            pass  # Some pages have no extractable tables

        # 2. Extract prose text (suppress table bounding boxes to avoid duplication)
        try:
            prose = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        except Exception:
            prose = ""

        # Remove table content we already captured (rough dedup)
        for tt in table_texts:
            # Remove first line of the table (header) from prose if present
            header = tt.split("\n")[0]
            prose = prose.replace(header, "")

        prose = prose.strip()
        if not prose:
            return chunks

        # 3. Chunk the prose with a sliding window
        for window in _sliding_window_chunks(prose, self.chunk_size, self.overlap):
            window = window.strip()
            if not window:
                continue
            tags = _detect_section_tags(window)
            chunks.append(
                Chunk(
                    text=window,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "type": "text",
                        "tags": tags,
                    },
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ingest(self, pdf_path: str | Path) -> list[Chunk]:
        """
        Parse *pdf_path* and return all chunks extracted from it.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks with metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source = pdf_path.name
        all_chunks: list[Chunk] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            print(f"[ingestor] Opening '{source}' — {total} pages")
            for i, page in enumerate(pdf.pages, start=1):
                page_chunks = self._process_page(page, source, i)
                all_chunks.extend(page_chunks)
                if i % 10 == 0 or i == total:
                    print(f"[ingestor]   Processed {i}/{total} pages "
                          f"({len(all_chunks)} chunks so far)")

        print(f"[ingestor] Done. Total chunks: {len(all_chunks)}")
        return all_chunks