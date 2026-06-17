"""
ingestor.py
-----------
Parses insurance policy PDFs using pymupdf4llm.
Leverages Markdown structure for intelligent, hierarchy-aware chunking.
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

def _tags(text: str) -> list[str]:
    return list({m.group(0).strip().lower() for m in _TAG_RE.finditer(text)})

# ---------------------------------------------------------------------------
# Public ingestor
# ---------------------------------------------------------------------------

class PDFIngestor:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
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
        # Character splitter ensures no single chunk exceeds context limits
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

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
            # Ensure we capture the native page number from the extraction
            page_num = page_data.get("metadata", {}).get("page", i - 1) + 1  
            
            if not page_text.strip():
                continue

            # 1. Split by Markdown Headers
            header_splits = self.md_splitter.split_text(page_text)

            for split in header_splits:
                # Combine headers to form the structural heading/clause
                heading_parts = [v for k, v in split.metadata.items() if k.startswith("Header")]
                heading = " - ".join(heading_parts) if heading_parts else ""

                # Define the parent text (useful for extended context in chain.py)
                parent_text = split.page_content
                if len(parent_text.split()) > 2000:
                    parent_text = " ".join(parent_text.split()[:2000]) + "..."

                # 2. Sub-split large sections into strict sizes
                sub_splits = self.text_splitter.split_text(split.page_content)

                for sub_text in sub_splits:
                    if len(sub_text.split()) < 10:  # Ignore tiny artifact chunks
                        continue

                    tags = _tags(sub_text)
                    is_definition = "definition" in heading.lower() or "Def." in sub_text
                    if is_definition and "definition" not in tags:
                        tags.append("definition")
                    if "critical illness" in heading.lower() and "critical_illness" not in tags:
                        tags.append("critical_illness")

                    chunk = Chunk(
                        text=sub_text,
                        metadata={
                            "source": source,
                            "page": page_num,
                            "line": 1,  # Line numbers are less precise in MD, defaulting to 1
                            "clause": heading,
                            "heading": heading,
                            "type": "text",
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