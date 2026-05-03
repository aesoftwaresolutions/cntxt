from dataclasses import dataclass, field
from datetime import datetime
import json
import pathlib
from typing import Any, Optional

from src.core.chunker import Chunk


@dataclass
class IndexState:
    last_indexed: dict[str, datetime] = field(default_factory=dict)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {doc_id: ts.isoformat() for doc_id, ts in self.last_indexed.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path) -> "IndexState":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(last_indexed={k: datetime.fromisoformat(v) for k, v in data.items()})


class DocumentIndexer:
    def __init__(
        self,
        connector: Any,
        chunker: Any,
        retriever: Any,
        state_path: pathlib.Path,
    ) -> None:
        self.connector = connector
        self.chunker = chunker
        self.retriever = retriever
        self.state_path = state_path
        self.state = IndexState.load(state_path)

    def index_all(self, folder_id: Optional[str] = None) -> int:
        docs = self.connector.list_documents(folder_id=folder_id)
        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self._index_doc(doc)
            all_chunks.extend(chunks)
            self.state.last_indexed[doc.id] = datetime.now()
        self.retriever.index(all_chunks)
        self.state.save(self.state_path)
        return len(all_chunks)

    def index_incremental(self, folder_id: Optional[str] = None) -> int:
        docs = self.connector.list_documents(folder_id=folder_id)
        earliest = self._earliest_checkpoint()
        changed = [d for d in docs if earliest is None or d.modified_at > earliest]
        all_chunks: list[Chunk] = []
        for doc in changed:
            chunks = self._index_doc(doc)
            all_chunks.extend(chunks)
            self.state.last_indexed[doc.id] = datetime.now()
        self.retriever.index(all_chunks)
        self.state.save(self.state_path)
        return len(all_chunks)

    def _index_doc(self, doc: Any) -> list[Chunk]:
        text = self.connector.read_document(doc)
        return self.chunker.chunk_document(doc_id=doc.id, doc_name=doc.name, text=text)

    def _earliest_checkpoint(self) -> Optional[datetime]:
        if not self.state.last_indexed:
            return None
        return min(self.state.last_indexed.values())
