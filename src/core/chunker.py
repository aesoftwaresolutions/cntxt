import uuid
from dataclasses import dataclass, field
from typing import Optional
import tiktoken


@dataclass
class Chunk:
    """Represents a document chunk with contextual information."""
    id: str
    doc_id: str
    doc_name: str
    text: str
    raw_text: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)


class ContextualChunker:
    """Chunks documents and prepends document-level context before embedding (Anthropic's Contextual Retrieval method)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, context_window: int = 150):
        """Initialize chunker with token-based sizing and context window."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, doc_id: str, doc_name: str, text: str) -> list[Chunk]:
        """Split document into overlapping chunks with prepended document context."""
        chunks_list = self._split_tokens(text, self.chunk_size, self.chunk_overlap)

        # Extract document summary (first context_window tokens)
        context_tokens = self.tokenizer.encode(text)[:self.context_window]
        context_summary = self.tokenizer.decode(context_tokens)

        result = []
        total_chunks = len(chunks_list)

        for idx, raw_chunk in enumerate(chunks_list):
            # Prepend context to chunk
            contextual_text = f"[Doc: {doc_name}] {context_summary}\n\n{raw_chunk}"

            chunk = Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_name=doc_name,
                text=contextual_text,
                raw_text=raw_chunk,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata={}
            )
            result.append(chunk)

        return result

    def _split_tokens(self, text: str, size: int, overlap: int) -> list[str]:
        """Split text into overlapping token windows using tiktoken."""
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []

        chunks = []
        step = max(1, size - overlap)
        start = 0

        while start < len(tokens):
            end = min(start + size, len(tokens))
            chunks.append(self.tokenizer.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step

        return chunks
