import numpy as np
from rank_bm25 import BM25Okapi
from .chunker import Chunk
from .embedder import Embedder
from typing import Optional


class HybridRetriever:
    """Combines vector cosine similarity with BM25 using RRF. Embedder is optional — omit for BM25-only mode."""

    def __init__(self, embedder: Optional[Embedder] = None, alpha: float = 0.6):
        self.embedder = embedder
        self.alpha = alpha
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: list[Chunk] = []
        self.bm25_index: Optional[BM25Okapi] = None

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index; also embed if an embedder is configured."""
        self.chunks = chunks

        self.tokenized_texts = [chunk.raw_text.lower().split() for chunk in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_texts) if chunks else None

        if self.embedder is not None and chunks:
            embeddings_list = self.embedder.embed_texts([c.text for c in chunks])
            self.embeddings = np.array(embeddings_list)
        else:
            self.embeddings = None

    def query(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Retrieve top_k chunks via hybrid RRF, or pure BM25 if no embedder."""
        if not self.chunks or self.bm25_index is None:
            return []

        bm25_scores = np.array(self.bm25_index.get_scores(query.lower().split()))

        if self.embedder is not None and self.embeddings is not None:
            query_vec = np.array(self.embedder.embed_query(query))
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
            vector_scores = np.dot(self.embeddings, query_vec) / norms

            fused = np.zeros(len(self.chunks))
            for rank, idx in enumerate(np.argsort(-vector_scores)):
                fused[idx] += self.alpha / (rank + 60)
            for rank, idx in enumerate(np.argsort(-bm25_scores)):
                fused[idx] += (1 - self.alpha) / (rank + 60)
        else:
            fused = bm25_scores

        top_indices = np.argsort(-fused)[:top_k]
        return [(self.chunks[i], float(fused[i])) for i in top_indices]
