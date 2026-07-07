"""
ingestor.py
-----------
Parses insurance policy PDFs using pymupdf4llm.
Leverages Markdown structure for intelligent, hierarchy-aware chunking.

Tables are treated as first-class, atomic units: a markdown table is never
split mid-row by the prose splitter, and its header row is always preserved
so a retrieved chunk is self-describing (e.g. a "Sum Insured / Room Rent /
Copay" row still carries those column labels even if it's the only chunk
the retriever returns).
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from datatypes import Chunk

# ---------------------------------------------------------------------------
# Regex patterns for semantic tagging
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(
    r"waiting period|co[\-\s]?pay|sub[\-\s]?limit|exclusion|pre[\-\s]?existing"
    r"|deductible|premium|coverage|covered|network hospital|claim|grace period"
    r"|renewal|sum insured|room rent|icu|day care|maternity|ambulance"
    r"|no claim bonus|ncb|portability|free look",
    re.IGNORECASE,
)

# Matches a contiguous block of markdown table lines:
#   |  col | col | col  |
#   |  --- | --- | ---  |
#   |  val | val | val  |
# A table block is a run of 2+ lines that each start with "|" (allowing
# leading whitespace), where the second line is a header-separator row
# (only -, :, |, and whitespace).
_TABLE_BLOCK_RE = re.compile(
    r"(?:^[ \t]*\|.*\|[ \t]*\n)"        # header row
    r"(?:^[ \t]*\|[\s\-:|]*\|[ \t]*\n)"  # separator row (---|---|---)
    r"(?:^[ \t]*\|.*\|[ \t]*\n?)*",      # data rows
    re.MULTILINE,
)

_ROW_SPLIT_RE = re.compile(r"\n(?=[ \t]*\|)")


def _tags(text: str) -> list[str]:
    return list({m.group(0).strip().lower() for m in _TAG_RE.finditer(text)})


def _split_text_around_tables(text: str) -> list[tuple[str, bool]]:
    """
    Split markdown text into an ordered list of (segment, is_table) tuples.
    Table segments are returned verbatim and untouched; everything else is
    plain prose that the caller can still run through normal splitting.
    """
    segments: list[tuple[str, bool]] = []
    pos = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        start, end = match.span()
        if start > pos:
            prose = text[pos:start]
            if prose.strip():
                segments.append((prose, False))
        table_text = match.group(0).rstrip("\n")
        if table_text.strip():
            segments.append((table_text, True))
        pos = end
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            segments.append((tail, False))
    return segments


def _chunk_table(
    table_text: str,
    max_chars: int,
) -> list[str]:
    """
    Keep a markdown table intact as a single chunk whenever possible.
    Only if the table is larger than max_chars do we split it — and when we
    do, the header row + separator row are repeated on every resulting
    piece so each remains self-describing in isolation.
    """
    if len(table_text) <= max_chars:
        return [table_text]

    rows = _ROW_SPLIT_RE.split(table_text)
    rows = [r for r in rows if r.strip()]
    if len(rows) < 3:
        # Pathologically wide single row/header — nothing sensible to split.
        return [table_text]

    header, separator, data_rows = rows[0], rows[1], rows[2:]
    prefix = header + "\n" + separator + "\n"

    pieces: list[str] = []
    current = prefix
    for row in data_rows:
        candidate = current + row + "\n"
        if len(candidate) > max_chars and current != prefix:
            pieces.append(current.rstrip("\n"))
            current = prefix + row + "\n"
        else:
            current = candidate
    if current.strip() != prefix.strip():
        pieces.append(current.rstrip("\n"))

    return pieces or [table_text]


# ---------------------------------------------------------------------------
# Public ingestor
# ---------------------------------------------------------------------------

class PDFIngestor:
    def __init__(self, chunk_size: int = 800, overlap: int = 100, table_max_chars: int | None = None):
        # Markdown splitter keeps contextual headers attached to the chunks
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False,
        )
        # Character splitter ensures no single PROSE chunk exceeds context limits.
        # Tables are never passed through this splitter (see _chunk_table).
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunk_size = chunk_size
        # Tables get a more generous ceiling than prose by default, since
        # splitting a table is much more destructive than splitting prose
        # and we'd rather keep it whole if it's even somewhat oversized.
        self.table_max_chars = table_max_chars or max(chunk_size * 3, 3000)

    def ingest(self, pdf_path: str | Path) -> list[Chunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source = pdf_path.name
        all_chunks: list[Chunk] = []

        print(f"[ingestor] '{source}' — Converting to Markdown...")

        # Extract page by page to maintain page number metadata
        try:
            md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF with pymupdf4llm: {e}")

        total = len(md_pages)
        print(f"[ingestor] '{source}' — {total} pages extracted. Chunking...")

        for i, page_data in enumerate(md_pages, start=1):
            page_text = page_data.get("text", "")

            # pymupdf4llm's page_chunks metadata exposes the page number under
            # "page_number" (already 1-indexed). Older/newer versions of the
            # library have used different key names, so we check a couple of
            # known variants before falling back to the loop index.
            page_meta = page_data.get("metadata", {}) or {}
            page_num = (
                page_meta.get("page_number")
                if page_meta.get("page_number") is not None
                else page_meta.get("page")
            )
            if page_num is None:
                page_num = i  # loop index is already 1-indexed via start=1

            if not page_text.strip():
                continue

            # 1. Split by Markdown Headers (this also walks past tables fine,
            #    since headers are "#" lines, not "|" lines)
            header_splits = self.md_splitter.split_text(page_text)

            for split in header_splits:
                # Combine headers to form the structural heading/clause
                heading_parts = [v for k, v in split.metadata.items() if k.startswith("Header")]
                heading = " - ".join(heading_parts) if heading_parts else ""

                # Define the parent text (useful for extended context in chain.py)
                parent_text = split.page_content
                if len(parent_text.split()) > 2000:
                    parent_text = " ".join(parent_text.split()[:2000]) + "..."

                # 2. Separate tables from prose BEFORE running the character
                #    splitter, so tables are never sliced mid-row.
                segments = _split_text_around_tables(split.page_content)

                for segment_text, is_table in segments:
                    if is_table:
                        sub_texts = _chunk_table(segment_text, self.table_max_chars)
                    else:
                        sub_texts = self.text_splitter.split_text(segment_text)

                    for sub_text in sub_texts:
                        word_count = len(sub_text.split())
                        # Ignore tiny artifact chunks, but never drop a table
                        # purely for being "short" — a 3-row table can be
                        # under 10 words and still be the entire answer.
                        if word_count < 10 and not is_table:
                            continue

                        tags = _tags(sub_text)
                        is_definition = "definition" in heading.lower() or "Def." in sub_text
                        if is_definition and "definition" not in tags:
                            tags.append("definition")
                        if "critical illness" in heading.lower() and "critical_illness" not in tags:
                            tags.append("critical_illness")
                        if is_table:
                            tags.append("table")

                        chunk = Chunk(
                            text=sub_text,
                            metadata={
                                "source": source,
                                "page": page_num,
                                "line": 1,  # Line numbers are less precise in MD, defaulting to 1
                                "clause": heading,
                                "heading": heading,
                                "type": "table" if is_table else "text",
                                "tags": tags,
                                "parent_text": parent_text,
                                "is_definition": is_definition,
                            },
                        )
                        all_chunks.append(chunk)

            if i % 10 == 0 or i == total:
                print(f"[ingestor]   {i}/{total} pages processed | {len(all_chunks)} chunks generated")

        print(f"[ingestor] Done. {len(all_chunks)} total chunks.")
        return all_chunks