

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestor import PDFIngestor
from vector_store import PolicyVectorStore

CHROMA_DIR = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Folder containing your fixed policy PDFs
DOCUMENT_DIR = Path("static/documents")


# PDF filename -> policy identifier
POLICIES = {
    "hdfc health plan.pdf": "hdfc_health",
    "icici lombard plan.pdf": "icici_health",
    "LIC insurance.pdf": "lic_health",
}


def main():

    store = PolicyVectorStore(persist_dir=CHROMA_DIR)

    ingestor = PDFIngestor(
        chunk_size=600,
        overlap=100
    )

    total_added = 0

    for filename, policy_id in POLICIES.items():

        pdf_path = DOCUMENT_DIR / filename

        if not pdf_path.exists():
            print(f"[ERROR] Missing file: {pdf_path}")
            continue

        print("\n" + "=" * 60)
        print(f"Ingesting: {policy_id}")
        print("=" * 60)

        chunks = ingestor.ingest(
            pdf_path,
            policy_id
        )

        added = store.add_chunks(chunks)

        total_added += added

        print(
            f"[DONE] {policy_id}: {added} chunks added"
        )


    print("\n" + "=" * 60)
    print(
        f"Finished. Total chunks added: {total_added}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()