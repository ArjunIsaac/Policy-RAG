from pathlib import Path
from typing import List, Dict
import fitz


class PDFReader:
    """
    Reads a PDF while preserving:
    - page numbers
    - metadata
    - layout order
    """

    def __init__(self):
        pass

    def read(self, pdf_path: str | Path) -> List[Dict]:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        document = fitz.open(pdf_path)

        pages = []

        for page_index in range(len(document)):

            page = document.load_page(page_index)

            text = page.get_text("text", sort=True)

            blocks = page.get_text("blocks")

            pages.append(
                {
                    "page": page_index + 1,
                    "text": text.strip(),
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": blocks
                }
            )

        document.close()

        return pages