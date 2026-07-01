from app.embeddings.embedder import Embedder
from app.retrieval.chroma_store import ChromaStore

# -------------------------
# INIT
# -------------------------
embedder = Embedder()
store = ChromaStore()

# -------------------------
# TEST QUERY
# -------------------------
query = "What is the grace period?"

# 1. embed query
query_embedding = embedder.encode(query)

# 2. search
results = store.search(
    query_embedding=query_embedding,
    top_k=5
)

# 3. print results
for i, r in enumerate(results):

    print("\n--- RESULT", i + 1, "---")
    print("SCORE:", r["score"])
    print("PAGE:", r["page"])
    print("HEADING:", r["heading"])
    print("TEXT:", r["text"][:300])