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

_TABLE_BLOCK_RE = re.compile(
    r"(?:^[ \t]*\|.*\|[ \t]*\n)"        # header row
    r"(?:^[ \t]*\|[\s\-:|]*\|[ \t]*\n)"  # separator row (---|---|---)
    r"(?:^[ \t]*\|.*\|[ \t]*\n?)*",      # data rows
    re.MULTILINE,
)

_ROW_SPLIT_RE = re.compile(r"\n(?=[ \t]*\|)")


def _tags(text: str) -> list[str]:
    return list({m.group(0).strip().lower() for m in _TAG_RE.finditer(text)})


from typing import Literal

SegmentType = Literal["prose", "table", "table_context"]


def _split_text_around_tables(
    text: str,
    context_before: int = 350,
    context_after: int = 200,
) -> list[tuple[str, SegmentType]]:


    segments: list[tuple[str, SegmentType]] = []

    pos = 0

    for match in _TABLE_BLOCK_RE.finditer(text):
        start, end = match.span()

        #
        # Prose before the table
        #
        if start > pos:
            prose = text[pos:start].strip()

            if prose:
                segments.append((prose, "prose"))

        #
        # Raw table
        #
        table_text = match.group(0).rstrip("\n")

        if table_text.strip():

            segments.append((table_text, "table"))

            #
            # Contextual table
            #
            before_start = max(0, start - context_before)
            after_end = min(len(text), end + context_after)

            before = text[before_start:start].strip()
            after = text[end:after_end].strip()

            contextual_chunk = "\n\n".join(
                part
                for part in (before, table_text, after)
                if part.strip()
            )

            segments.append((contextual_chunk, "table_context"))

        pos = end

    #
    # Remaining prose
    #
    if pos < len(text):
        prose = text[pos:].strip()

        if prose:
            segments.append((prose, "prose"))

    return segments


def build_table_embedding_text(
    heading: str,
    table_text: str,
    segment_type: str = "table",
) -> str:
    """
    Produce a semantic representation of a markdown table.

    The original markdown is preserved, but we prepend a
    natural-language description that embedding models understand
    much better.
    """

    lines = [l.rstrip() for l in table_text.splitlines() if l.strip()]

    if len(lines) < 2:
        return table_text

    header = lines[0]

    columns = [
        c.strip("* ").strip()
        for c in header.strip("|").split("|")
    ]

    data_rows = []

    for row in lines[2:]:
        cells = [
            c.strip("* ").strip()
            for c in row.strip("|").split("|")
        ]

        if any(cells):
            data_rows.append(cells)

    row_count = len(data_rows)

    preview = []

    for row in data_rows[:8]:
        preview.append(", ".join(row))

    preview_text = "\n".join(f"- {r}" for r in preview)

    key_entities = []

    for row in data_rows[:20]:
        if row:
            key_entities.append(row[0])

    key_entities = list(dict.fromkeys(key_entities))


    if segment_type == "table":
        summary = (
            "This chunk contains the extracted table itself. "
            "Each row represents a structured record."
        )
    elif segment_type == "table_context":
        summary = (
            "This chunk contains a structured table together with the "
            "surrounding explanatory text from the same section."
        )

    description = f"""
    Heading:
    {heading or "Untitled Section"}

    Content Type:
    Structured Table

    Summary:
    {summary}

    Columns:
    {", ".join(columns)}

    Number of Rows:
    {row_count}


    Key Entities:
    {", ".join(key_entities)}


    Example Records:
    {preview_text}

    Original Markdown Table:

    """

    return description + table_text

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
            print(type(md_pages))

            DEBUG_SAVE_MARKDOWN = True

            if DEBUG_SAVE_MARKDOWN:

                with open("debug_markdown.md", "w", encoding="utf-8") as f:

                    if isinstance(md_pages, str):
                        f.write(md_pages)

                    else:
                        for i, page in enumerate(md_pages):
                            f.write("=" * 80 + "\n")
                            f.write(f"PAGE {i+1}\n")
                            f.write("=" * 80 + "\n\n")

                            # dump everything in the dict
                            for key, value in page.items():
                                f.write(f"## {key}\n")
                                f.write(str(value))
                                f.write("\n\n")

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

                for segment_text, segment_type in segments:
                    if segment_type in ("table", "table_context"):
                        sub_texts = _chunk_table(segment_text, self.table_max_chars)
                    else:
                        sub_texts = self.text_splitter.split_text(segment_text)

                    for sub_text in sub_texts:
                        word_count = len(sub_text.split())

                        if word_count < 10 and not segment_type.startswith("table"):
                            continue

                        tags = _tags(sub_text)
                        is_definition = "definition" in heading.lower() or "Def." in sub_text
                        if is_definition and "definition" not in tags:
                            tags.append("definition")
                        if "critical illness" in heading.lower() and "critical_illness" not in tags:
                            tags.append("critical_illness")
                        if segment_type == "table" and "table" not in tags:
                            tags.append("table")

                        if segment_type.startswith("table"):
                            embedding_text = build_table_embedding_text(heading=heading, table_text=sub_text,segment_type=segment_type)

                        else:
                            embedding_text = (f"Heading: {heading}\n\n{sub_text}" if heading else sub_text)
                        chunk = Chunk(
                            text=sub_text,
                            metadata={
                                "source": source,
                                "page": page_num,
                                "line": 1,  # Line numbers are less precise in MD, defaulting to 1
                                "clause": heading,
                                "heading": heading,
                                "type": segment_type,
                                "tags": tags,
                                "parent_text": parent_text,
                                "is_definition": is_definition,
                                "embedding_text": embedding_text,
                                
                            },
                        )
                        all_chunks.append(chunk)

            if i % 10 == 0 or i == total:
                print(f"[ingestor]   {i}/{total} pages processed | {len(all_chunks)} chunks generated")

        print(f"[ingestor] Done. {len(all_chunks)} total chunks.")
        return all_chunks