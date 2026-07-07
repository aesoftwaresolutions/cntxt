from datetime import datetime
from typing import Any, Optional

import numpy as np

from src.core.chunker import Chunk
from src.indexing.store import IndexState, IndexStore

__all__ = ["DocumentIndexer", "IndexState", "IndexStore"]


class DocumentIndexer:
    def __init__(
        self,
        connector: Any,
        chunker: Any,
        retriever: Any,
        store: IndexStore,
    ) -> None:
        self.connector = connector
        self.chunker = chunker
        self.retriever = retriever
        self.store = store

        self.state, chunks, embeddings = self.store.load()
        if chunks:
            self.retriever.load(chunks, embeddings)

    def index_all(self, folder_id: Optional[str] = None) -> int:
        docs = self.connector.list_documents(folder_id=folder_id)
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self._index_doc(doc))
            self.state.last_indexed[doc.id] = datetime.now()
        self.retriever.index(all_chunks)
        self._persist()
        return len(all_chunks)

    def index_incremental(self, folder_id: Optional[str] = None) -> int:
        docs = self.connector.list_documents(folder_id=folder_id)
        earliest = self._earliest_checkpoint()
        changed = [d for d in docs if earliest is None or d.modified_at > earliest]
        changed_ids = {d.id for d in changed}

        # Keep chunks (and their embeddings) for docs that didn't change, so a poll
        # with no changes doesn't wipe out everything indexed so far.
        kept_chunks = [c for c in self.retriever.chunks if c.doc_id not in changed_ids]
        kept_embeddings = self._embeddings_for(kept_chunks)

        new_chunks: list[Chunk] = []
        for doc in changed:
            new_chunks.extend(self._index_doc(doc))
            self.state.last_indexed[doc.id] = datetime.now()

        self.retriever.merge(kept_chunks, kept_embeddings, new_chunks)
        self._persist()
        return len(new_chunks)

    def _index_doc(self, doc: Any) -> list[Chunk]:
        text = self.connector.read_document(doc)
        return self.chunker.chunk_document(doc_id=doc.id, doc_name=doc.name, text=text)

    def _earliest_checkpoint(self) -> Optional[datetime]:
        if not self.state.last_indexed:
            return None
        return min(self.state.last_indexed.values())

    def _embeddings_for(self, chunks: list[Chunk]) -> Optional[np.ndarray]:
        if self.retriever.embeddings is None or not chunks:
            return None
        row_by_id = {c.id: i for i, c in enumerate(self.retriever.chunks)}
        rows = [row_by_id[c.id] for c in chunks]
        return self.retriever.embeddings[rows]

    def _persist(self) -> None:
        self.store.save(self.state, self.retriever.chunks, self.retriever.embeddings)
