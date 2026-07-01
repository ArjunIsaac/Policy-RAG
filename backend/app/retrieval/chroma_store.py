import chromadb


class ChromaStore:

    def __init__(self, persist_dir: str = "./chroma_db"):

        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.collection = self.client.get_or_create_collection(
            name="insurance_policies"
        )

    # -------------------------
    # ADD DOCUMENTS
    # -------------------------
    def add_chunks(self, chunks: list[dict]):

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
            embeddings.append(chunk["embedding"])

            metadatas.append({
                "page": chunk.get("page"),
                "heading": chunk.get("heading"),
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk.get("document_id", "current_doc")
            })

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    # -------------------------
    # QUERY
    # -------------------------
    def search(
        self,
        query_embedding: list,
        top_k: int = 5,
        document_id: str | None = None
    ):

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if document_id is not None:
            kwargs["where"] = {
                "document_id": document_id
            }

        results = self.collection.query(**kwargs)

        cleaned = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):

            cleaned.append({
                "text": doc,
                "page": meta.get("page"),
                "heading": meta.get("heading"),
                "chunk_id": meta.get("chunk_id"),
                "document_id": meta.get("document_id"),
                "score": 1 - distance
            })

        return cleaned

    # -------------------------
    # CLEAR DATABASE
    # -------------------------
    def clear(self):

        try:
            self.client.delete_collection("insurance_policies")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="insurance_policies"
        )

    # -------------------------
    # COUNT CHUNKS
    # -------------------------
    def count(self):

        return self.collection.count()