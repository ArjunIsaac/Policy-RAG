from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os

from app.ingestion.pdf_reader import PDFReader
from app.ingestion.semantic_chunker import SemanticChunker
from app.embeddings.embedder import Embedder
from app.retrieval.chroma_store import ChromaStore
from app.extraction.attribute_engine import AttributeEngine, ATTRIBUTE_QUESTIONS

from app.db.database import Base, engine
from app.db import models  # IMPORTANT: ensures tables are registered

app = FastAPI()

# -------------------------
# CREATE DB TABLES ON STARTUP
# -------------------------
Base.metadata.create_all(bind=engine)

# -------------------------
# GLOBAL STATE
# -------------------------
UPLOAD_DIR = "uploads"
CURRENT_FILE = None
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# INIT COMPONENTS
# -------------------------
reader = PDFReader()
chunker = SemanticChunker()
embedder = Embedder()
store = ChromaStore()
engine = AttributeEngine()


# -------------------------
# 1. UPLOAD + INGEST
# -------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    global CURRENT_FILE

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    CURRENT_FILE = file_path

    # 1. READ PDF
    pages = reader.read(file_path)

    # 2. CHUNK
    chunks = chunker.chunk(pages)

    # 3. EMBED
    texts = [c["text"] for c in chunks]
    embeddings = embedder.batch_encode(texts)

    # 4. ATTACH EMBEDDINGS
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{i:05d}"
        chunk["embedding"] = embeddings[i]

    # 5. CLEAR PREVIOUS DOCUMENT
    store.clear()

    # 6. STORE NEW DOCUMENT
    store.add_chunks(chunks)

    return {
        "status": "success",
        "pages": len(pages),
        "chunks": len(chunks)
    }


# -------------------------
# 2. SINGLE ATTRIBUTE EXTRACTION
# -------------------------
@app.get("/extract/{attribute}")
def extract_attribute(attribute: str):

    if CURRENT_FILE is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    try:
        pages = reader.read(CURRENT_FILE)
        result = engine.extract(attribute, pages=pages)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# 3. FULL POLICY EXTRACTION
# -------------------------
@app.get("/extract/all")
def extract_all():

    if CURRENT_FILE is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    pages = reader.read(CURRENT_FILE)

    results = {}

    for attr in ATTRIBUTE_QUESTIONS.keys():
        try:
            results[attr] = engine.extract(attr, pages=pages)
        except Exception as e:
            results[attr] = {"error": str(e)}

    return {
        "status": "complete",
        "results": results
    }