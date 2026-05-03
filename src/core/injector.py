import tiktoken
from .chunker import Chunk


class ContextInjector:
    """Formats retrieved chunks into a context block and prepends to the user's prompt."""

    def __init__(self, max_tokens: int = 4000, separator: str = "---"):
        """Initialize injector with token limit and separator."""
        self.max_tokens = max_tokens
        self.separator = separator
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def build_context_block(self, results: list[tuple[Chunk, float]]) -> str:
        """Format retrieved chunks into a context block, truncated to max_tokens."""
        formatted_chunks = []

        for chunk, score in results:
            header = f"[Source: {chunk.doc_name}, chunk {chunk.chunk_index + 1}/{chunk.total_chunks}]"
            formatted = f"{header}\n{chunk.raw_text}"
            formatted_chunks.append(formatted)

        # Join with separator
        context_block = f"\n{self.separator}\n".join(formatted_chunks)

        # Truncate to max_tokens
        tokens = self.tokenizer.encode(context_block)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            context_block = self.tokenizer.decode(tokens)

        return context_block

    def inject(self, prompt: str, results: list[tuple[Chunk, float]]) -> str:
        """Prepend context block to prompt in <context> tags."""
        context_block = self.build_context_block(results)
        return f"<context>\n{context_block}\n</context>\n\n{prompt}"
