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
        """Build BM25 index from scratch; also embed if an embedder is configured."""
        self.chunks = chunks
        self._rebuild_bm25()
        self.embeddings = self._embed_all(chunks) if self.embedder is not None and chunks else None

    def load(self, chunks: list[Chunk], embeddings: Optional[np.ndarray]) -> None:
        """Hydrate from a previously persisted index (see IndexStore) without recomputing embeddings."""
        self.chunks = chunks
        self._rebuild_bm25()
        self.embeddings = embeddings

    def merge(
        self,
        kept_chunks: list[Chunk],
        kept_embeddings: Optional[np.ndarray],
        new_chunks: list[Chunk],
    ) -> None:
        """Keep already-embedded chunks as-is and only embed the newly (re)indexed ones."""
        self.chunks = kept_chunks + new_chunks
        self._rebuild_bm25()

        if self.embedder is None or not self.chunks:
            self.embeddings = None
            return

        new_embeddings = self._embed_all(new_chunks) if new_chunks else None
        if kept_embeddings is not None and new_embeddings is not None:
            self.embeddings = np.vstack([kept_embeddings, new_embeddings])
        else:
            self.embeddings = kept_embeddings if kept_embeddings is not None else new_embeddings

    def _rebuild_bm25(self) -> None:
        self.tokenized_texts = [chunk.raw_text.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(self.tokenized_texts) if self.chunks else None

    def _embed_all(self, chunks: list[Chunk]) -> np.ndarray:
        embeddings_list = self.embedder.embed_texts([c.text for c in chunks])
        return np.array(embeddings_list)

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
