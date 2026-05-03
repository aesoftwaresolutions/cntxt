from pathlib import Path
from datetime import datetime
import mimetypes
from .base import Connector, DocumentMeta

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from pdfminer.high_level import extract_text as extract_pdf_text
except ImportError:
    extract_pdf_text = None


class iCloudConnector(Connector):
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".py", ".js", ".ts", ".json",
        ".yaml", ".yml", ".csv", ".docx", ".pdf",
    }
    EXCLUDED_DIRS = {
        ".venv", "venv", "env", ".env",
        "__pycache__", ".git", ".idea", ".vscode",
        "node_modules", "dist", "build", ".eggs", "*.egg-info",
    }

    def __init__(self, root: str | None = None, folders: list[str] | None = None):
        self.root = Path(root or "C:/Users/kirk7/iCloudDrive")
        self.folders = folders or []

    def list_documents(self, folder_id: str | None = None) -> list[DocumentMeta]:
        if folder_id:
            roots = [self.root / folder_id.lstrip("/")]
        elif self.folders:
            roots = [self.root / f.lstrip("/") for f in self.folders]
        else:
            roots = [self.root]

        docs: list[DocumentMeta] = []
        for start in roots:
            for file_path in start.rglob("*"):
                if any(part in self.EXCLUDED_DIRS for part in file_path.parts):
                    continue
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    docs.append(self._file_to_meta(file_path))
        return docs

    def read_document(self, doc: DocumentMeta) -> str:
        file_path = Path(doc.path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {doc.path}")

        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv"}:
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix == ".docx":
            if DocxDocument is None:
                raise ImportError("python-docx is required to read .docx files")
            return "\n".join(p.text for p in DocxDocument(file_path).paragraphs)

        if suffix == ".pdf":
            if extract_pdf_text is None:
                raise ImportError("pdfminer.six is required to read .pdf files")
            return extract_pdf_text(str(file_path))

        raise ValueError(f"Unsupported file type: {suffix}")

    def get_changes_since(self, checkpoint: datetime) -> list[DocumentMeta]:
        return [d for d in self.list_documents() if d.modified_at > checkpoint]

    def supports_folders(self) -> bool:
        return True

    def _file_to_meta(self, file_path: Path) -> DocumentMeta:
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        relative = file_path.relative_to(self.root)
        return DocumentMeta(
            id=str(relative),
            name=file_path.name,
            path=str(file_path),
            source="icloud",
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            mime_type=mime_type or "application/octet-stream",
            size_bytes=stat.st_size,
        )
