import os
import pathlib
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5
    folder_id: Optional[str] = None


class SourceEntry(BaseModel):
    doc: str
    score: float


class QueryResponse(BaseModel):
    injected_prompt: str
    sources: list[SourceEntry]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


class ReindexResponse(BaseModel):
    chunks: int


class QueryDaemon:
    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = os.getenv("CNTXT_CONFIG", "config/config.yaml")

        self.config_path = pathlib.Path(config_path)
        self.config = self._load_config()

        self.app = FastAPI(title="CNTXT Daemon")
        self._setup_routes()

        self.indexer: Optional[Any] = None
        self.change_detector: Optional[Any] = None
        self.connector: Optional[Any] = None
        self.chunker: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.embedder: Optional[Any] = None
        self.injector: Optional[Any] = None

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_routes(self) -> None:
        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest) -> QueryResponse:
            if self.retriever is None or self.injector is None:
                raise HTTPException(status_code=503, detail="Not initialized — call /reindex first")

            results = self.retriever.query(query=request.prompt, top_k=request.top_k)
            sources = [SourceEntry(doc=chunk.doc_name, score=score) for chunk, score in results]
            injected = self.injector.inject(request.prompt, results)
            return QueryResponse(injected_prompt=injected, sources=sources)

        @self.app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            count = len(self.retriever.chunks) if self.retriever is not None else 0
            return HealthResponse(status="ok", chunks_indexed=count)

        @self.app.post("/reindex", response_model=ReindexResponse)
        async def reindex() -> ReindexResponse:
            if self.indexer is None:
                raise HTTPException(status_code=503, detail="Indexer not initialized")
            chunk_count = self.indexer.index_all()
            return ReindexResponse(chunks=chunk_count)

    def initialize(self) -> None:
        self._init_connectors()
        self._init_core()
        self._init_indexer()
        self._init_change_detector()

    def _init_connectors(self) -> None:
        from src.connectors import iCloudConnector, GoogleDriveConnector

        icloud_cfg = self.config.get("icloud", {})
        gdrive_cfg = self.config.get("google_drive", {})

        if icloud_cfg.get("enabled"):
            self.connector = iCloudConnector(
                root=icloud_cfg.get("root"),
                folders=icloud_cfg.get("folders", []),
            )
        elif gdrive_cfg.get("enabled"):
            self.connector = GoogleDriveConnector(
                credentials_file=gdrive_cfg.get("credentials_file"),
                folder_ids=gdrive_cfg.get("folder_ids", []),
            )

    def _init_core(self) -> None:
        from src.core import ContextualChunker, Embedder, HybridRetriever, ContextInjector

        chunk_cfg = self.config.get("chunking", {})
        retr_cfg = self.config.get("retrieval", {})
        ant_cfg = self.config.get("anthropic", {})

        self.chunker = ContextualChunker(
            chunk_size=chunk_cfg.get("chunk_size", 512),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
            context_window=chunk_cfg.get("context_window", 150),
        )
        self.embedder = Embedder(
            model=ant_cfg.get("embedding_model", "voyage-3"),
            batch_size=ant_cfg.get("batch_size", 64),
        )
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            alpha=retr_cfg.get("alpha", 0.6),
        )
        self.injector = ContextInjector(
            max_tokens=retr_cfg.get("max_tokens", 4000),
        )

    def _init_indexer(self) -> None:
        from src.indexing import DocumentIndexer

        state_path = self.config_path.parent / "index_state.json"
        self.indexer = DocumentIndexer(
            connector=self.connector,
            chunker=self.chunker,
            retriever=self.retriever,
            state_path=state_path,
        )

    def _init_change_detector(self) -> None:
        from src.indexing import ChangeDetector

        if self.indexer is None:
            return
        self.change_detector = ChangeDetector(indexer=self.indexer, poll_interval=300)
        self.change_detector.start()

    def run(self) -> None:
        self.initialize()
        srv = self.config.get("server", {})
        uvicorn.run(self.app, host=srv.get("host", "127.0.0.1"), port=srv.get("port", 8765))

    def shutdown(self) -> None:
        if self.change_detector is not None:
            self.change_detector.stop()


app = FastAPI(title="CNTXT")  # module-level app for uvicorn import


def main() -> None:
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    daemon = QueryDaemon(config_path=config_path)
    try:
        daemon.run()
    finally:
        daemon.shutdown()


if __name__ == "__main__":
    main()
