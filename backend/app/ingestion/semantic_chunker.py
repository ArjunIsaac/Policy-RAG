from typing import List, Dict

from app.ingestion.layout import LayoutDetector


class SemanticChunker:

    def __init__(
        self,
        max_chars: int = 1400
    ):

        self.max_chars = max_chars

        self.layout = LayoutDetector()

    def chunk(
        self,
        pages: List[Dict]
    ) -> List[Dict]:

        chunks = []

        current_heading = "Document"

        buffer = ""

        start_page = 1

        for page in pages:

            page_number = page["page"]

            lines = page["text"].splitlines()

            for line in lines:

                line = line.strip()

                if not line:

                    continue

                if self.layout.detect_heading(line):

                    if buffer:

                        chunks.append(
                            {
                                "heading": current_heading,
                                "page": start_page,
                                "text": buffer.strip(),
                            }
                        )

                    current_heading = line

                    buffer = ""

                    start_page = page_number

                    continue

                if len(buffer) + len(line) > self.max_chars:

                    chunks.append(
                        {
                            "heading": current_heading,
                            "page": start_page,
                            "text": buffer.strip(),
                        }
                    )

                    buffer = line

                    start_page = page_number

                else:

                    buffer += "\n" + line

        if buffer:

            chunks.append(
                {
                    "heading": current_heading,
                    "page": start_page,
                    "text": buffer.strip(),
                }
            )

        return chunks