from anthropic import Anthropic


class Embedder:
    """Generates embeddings using Anthropic's API (voyage-3 model)."""

    def __init__(self, model: str = "voyage-3", batch_size: int = 64):
        """Initialize embedder with model and batch size."""
        self.model = model
        self.batch_size = batch_size
        self.client = Anthropic()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with batching support."""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            for embedding_obj in response.embeddings:
                embeddings.append(embedding_obj.embedding)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.embeddings[0].embedding
