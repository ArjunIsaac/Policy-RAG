from app.embeddings.embedder import Embedder
from app.retrieval.chroma_store import ChromaStore


class Retriever:

    def __init__(self):

        self.embedder = Embedder()
        self.store = ChromaStore()

    # -------------------------
    # MAIN FUNCTION
    # -------------------------
    def retrieve(self, query: str, top_k: int = 5):

        # 1. Embed query
        query_embedding = self.embedder.encode(query)

        # 2. Search vector DB
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. Ensure sorted by similarity
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results