"""
ingestor.py
-----------
Parses insurance policy PDFs using pymupdf4llm.
Leverages Markdown structure for intelligent, hierarchy-aware chunking.
"""

from __future__ import annotations

import re
import unicodedata
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


def normalize_text(text: str) -> str:
    """
    Conservative text normalization for insurance policy documents.

    Safe operations only:
    - Unicode normalization
    - Remove control characters
    - Replace common ligatures
    - Normalize whitespace
    - Normalize quotes/dashes
    """

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace common ligatures
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    # Smart quotes → normal quotes
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

    # Long dashes
    text = (
        text.replace("–", "-")
            .replace("—", "-")
    )

    # Bullet variants
    text = (
        text.replace("•", "-")
            .replace("▪", "-")
            .replace("◦", "-")
    )

    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # Replace non-breaking spaces
    text = text.replace("\u00A0", " ")

    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()




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


import re

def enhance_document_structure(text: str) -> str:
    """
    Generic structure enhancement for any insurance policy.
    No hardcoded policy names or specific keywords.
    """
    
    # ===== 1. Detect and normalize numbered sections =====
    # Pattern: "1. Section Title" or "1) Section Title"
    section_pattern = re.compile(r'(?m)^(\d+)[\.\)]\s*([A-Z][A-Za-z\s]+)$')
    
    def format_section(match):
        num = match.group(1)
        title = match.group(2).strip()
        # Only if it looks like a meaningful section (not a short fragment)
        if len(title) > 5 and any(c.isupper() for c in title):
            return f"**Section {num}: {title}**\n\n"
        return match.group(0)
    
    text = section_pattern.sub(format_section, text)
    
    # ===== 2. Detect and normalize lists =====
    # Pattern: "i. item" or "a. item" or "1. item"
    # Replace with consistent "•" bullets for embedding
    def normalize_bullet(match):
        return "• " + match.group(1).strip()
    
    # Roman numerals (i., ii., iii., etc.)
    roman_pattern = re.compile(r'(?m)^\s*([ivx]+)\.\s*(.+)$')
    text = roman_pattern.sub(r'• \2', text)
    
    # Letters (a., b., c., etc.)
    letter_pattern = re.compile(r'(?m)^\s*([a-z])\.\s*(.+)$')
    text = letter_pattern.sub(r'• \2', text)
    
    # Numbers (1., 2., 3., etc.)
    number_pattern = re.compile(r'(?m)^\s*(\d+)\.\s*(.+)$')
    text = number_pattern.sub(r'• \2', text)
    
    # ===== 3. Detect "Exclusion" sections generically =====
    # Add structural markers around exclusion sections
    exclusion_pattern = re.compile(
        r'(?i)(exclusion|excluded|not cover|not payable|not indemnify)',
        re.MULTILINE
    )
    
    # Only add marker if it's at the start of a line or follows a heading
    lines = text.split('\n')
    enhanced_lines = []
    in_exclusion_section = False
    
    for line in lines:
        # Check if this line starts a heading with a section number
        heading_match = re.match(r'^\s*(\d+)\.?\s*([A-Z][a-zA-Z\s]+)', line)
        if heading_match:
            # Reset section tracking
            in_exclusion_section = False
        
        # Check if this line is about exclusions (at the start of a line)
        if re.match(r'(?i)^\s*(exclusion|not cover|not payable)', line):
            if not in_exclusion_section:
                in_exclusion_section = True
                enhanced_lines.append("## EXCLUSION SECTION\n")
        
        enhanced_lines.append(line)
    
    text = '\n'.join(enhanced_lines)
    
    # ===== 4. Add code identifiers to numbered sections =====
    # Generic: "Section 15" becomes "Section 15 - Code-SEC015"
    text = re.sub(
        r'(?m)^\*\*Section\s+(\d+):\s*(.+?)\*\*',
        r'**Section \1: \2 - Code-SEC\1**',
        text
    )
    
    return text


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

    def ingest(self, pdf_path: str | Path, policy_id:str= None) -> list[Chunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source = pdf_path.name
        print(f"[ingestor] Policy ID: {policy_id}")
        all_chunks: list[Chunk] = []

        print(f"[ingestor] '{source}' — Converting to Markdown...")

        # Extract page by page to maintain page number metadata
        try:
            md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            print(type(md_pages))

            DEBUG_SAVE_MARKDOWN = True
            if DEBUG_SAVE_MARKDOWN:

                # Create debug/ if it doesn't already exist
                debug_dir = Path("debug")
                debug_dir.mkdir(exist_ok=True)

                # One markdown file per PDF
                debug_file = debug_dir / f"{pdf_path.stem}.md"

                with open(debug_file, "w", encoding="utf-8") as f:

                    if isinstance(md_pages, str):
                        f.write(md_pages)

                    else:
                        for i, page in enumerate(md_pages):
                            f.write("=" * 80 + "\n")
                            f.write(f"PAGE {i+1}\n")
                            f.write("=" * 80 + "\n\n")

                            # Dump everything in the page dictionary
                            for key, value in page.items():
                                f.write(f"## {key}\n")
                                f.write(str(value))
                                f.write("\n\n")

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF with pymupdf4llm: {e}")

        total = len(md_pages)
        print(f"[ingestor] '{source}' — {total} pages extracted. Chunking...")

        for i, page_data in enumerate(md_pages, start=1):
            page_text = normalize_text(page_data.get("text", ""))

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

            header_splits = self.md_splitter.split_text(page_text)

            for split in header_splits:
                # Preserve the document hierarchy
                heading_path = [
                    value.strip()
                    for key, value in sorted(split.metadata.items())
                    if key.startswith("Header") and value.strip()
                ]

                heading = heading_path[-1] if heading_path else ""
                heading_full = " - ".join(heading_path)

                # Define the parent text (useful for extended context in chain.py)
                parent_text = normalize_text(split.page_content)
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
                        sub_text = normalize_text(sub_text)
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

                        embedding_text = normalize_text(embedding_text)

                        safe_policy_id = policy_id or Path(source).stem
                        chunk = Chunk(
                            text=sub_text,
                            metadata={
                                "source": source,
                                "policy_id": safe_policy_id,
                                "page": page_num,
                                "line": 1, 
                                "clause": heading,
                                "heading": heading,
                                "heading_full": heading_full,
                                "heading_path": " | ".join(heading_path) if heading_path else "",
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