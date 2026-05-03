import numpy as np
from rank_bm25 import BM25Okapi
from .chunker import Chunk
from .embedder import Embedder


class HybridRetriever:
    """Combines vector cosine similarity with BM25 keyword search using Reciprocal Rank Fusion."""

    def __init__(self, embedder: Embedder, alpha: float = 0.6):
        """Initialize retriever with embedder and fusion weight (alpha = vector weight)."""
        self.embedder = embedder
        self.alpha = alpha
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.bm25_index: BM25Okapi | None = None
        self.tokenized_texts: list[list[str]] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Embed chunks and build BM25 index over raw text."""
        self.chunks = chunks

        # Embed contextual text
        contextual_texts = [chunk.text for chunk in chunks]
        embeddings_list = self.embedder.embed_texts(contextual_texts)
        self.embeddings = np.array(embeddings_list)

        # Build BM25 index on raw text
        self.tokenized_texts = [chunk.raw_text.lower().split() for chunk in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_texts)

    def query(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Retrieve top_k results using hybrid fusion of vector and BM25 scores."""
        if self.embeddings is None or self.bm25_index is None:
            return []

        # Get vector scores
        query_embedding = np.array(self.embedder.embed_query(query))
        vector_scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )

        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Convert scores to ranks and apply RRF
        vector_ranks = np.argsort(-vector_scores)
        bm25_ranks = np.argsort(-bm25_scores)

        fused_scores = np.zeros(len(self.chunks))

        for rank, idx in enumerate(vector_ranks):
            fused_scores[idx] += self.alpha / (rank + 60)

        for rank, idx in enumerate(bm25_ranks):
            fused_scores[idx] += (1 - self.alpha) / (rank + 60)

        # Get top_k
        top_indices = np.argsort(-fused_scores)[:top_k]
        results = [(self.chunks[idx], float(fused_scores[idx])) for idx in top_indices]

        return results
