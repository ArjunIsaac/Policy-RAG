from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    _model = None

    def __init__(self):

        if Embedder._model is None:

            device = "cuda" if torch.cuda.is_available() else "cpu"

            Embedder._model = SentenceTransformer(
                "BAAI/bge-m3",
                device=device
            )

        self.model = Embedder._model

    def encode(self, text: str):
        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

    def batch_encode(self, texts, batch_size=32):

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            emb = self.model.encode(
                batch,
                normalize_embeddings=True
            )

            embeddings.extend(emb.tolist())

        return embeddings