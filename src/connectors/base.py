from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentMeta:
    id: str
    name: str
    path: str
    source: str
    modified_at: datetime
    mime_type: str
    size_bytes: int


class Connector(ABC):
    @abstractmethod
    def list_documents(self, folder_id: str | None = None) -> list[DocumentMeta]:
        pass

    @abstractmethod
    def read_document(self, doc: DocumentMeta) -> str:
        pass

    @abstractmethod
    def get_changes_since(self, checkpoint: datetime) -> list[DocumentMeta]:
        pass

    @abstractmethod
    def supports_folders(self) -> bool:
        pass
