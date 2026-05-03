#!/usr/bin/env python3
"""CNTXT — context retrieval daemon CLI."""

import argparse
import sys
from pathlib import Path


def _load_cfg(config_path: str) -> dict:
    import yaml
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file '{config_path}' not found", file=sys.stderr)
        sys.exit(1)


def handle_serve(args: argparse.Namespace) -> None:
    from src.server.daemon import QueryDaemon
    daemon = QueryDaemon(config_path=args.config)
    try:
        daemon.run()
    finally:
        daemon.shutdown()


def handle_query(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)

    from src.connectors import iCloudConnector, GoogleDriveConnector
    from src.core import ContextualChunker, Embedder, HybridRetriever, ContextInjector

    connector = None
    icloud_cfg = cfg.get("icloud", {})
    gdrive_cfg = cfg.get("google_drive", {})
    if icloud_cfg.get("enabled"):
        connector = iCloudConnector(
            root=icloud_cfg.get("root"),
            folders=icloud_cfg.get("folders", []),
        )
    elif gdrive_cfg.get("enabled"):
        connector = GoogleDriveConnector(
            credentials_file=gdrive_cfg.get("credentials_file"),
            folder_ids=gdrive_cfg.get("folder_ids", []),
        )

    if connector is None:
        print("Error: no connector enabled in config", file=sys.stderr)
        sys.exit(1)

    chunk_cfg = cfg.get("chunking", {})
    retr_cfg = cfg.get("retrieval", {})
    ant_cfg = cfg.get("anthropic", {})

    chunker = ContextualChunker(
        chunk_size=chunk_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
        context_window=chunk_cfg.get("context_window", 150),
    )
    embedder = Embedder(
        model=ant_cfg.get("embedding_model", "voyage-3"),
        batch_size=ant_cfg.get("batch_size", 64),
    )
    retriever = HybridRetriever(embedder=embedder, alpha=retr_cfg.get("alpha", 0.6))
    injector = ContextInjector(max_tokens=retr_cfg.get("max_tokens", 4000))

    docs = connector.list_documents(folder_id=args.folder)
    all_chunks = []
    for doc in docs:
        text = connector.read_document(doc)
        all_chunks.extend(chunker.chunk_document(doc.id, doc.name, text))

    retriever.index(all_chunks)
    results = retriever.query(args.prompt, top_k=args.top_k)
    print(injector.inject(args.prompt, results))


def handle_index(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)

    from src.connectors import iCloudConnector, GoogleDriveConnector
    from src.core import ContextualChunker, Embedder, HybridRetriever
    from src.indexing import DocumentIndexer

    icloud_cfg = cfg.get("icloud", {})
    gdrive_cfg = cfg.get("google_drive", {})
    chunk_cfg = cfg.get("chunking", {})
    retr_cfg = cfg.get("retrieval", {})
    ant_cfg = cfg.get("anthropic", {})

    connector = None
    if icloud_cfg.get("enabled"):
        connector = iCloudConnector(root=icloud_cfg.get("root"), folders=icloud_cfg.get("folders", []))
    elif gdrive_cfg.get("enabled"):
        connector = GoogleDriveConnector(
            credentials_file=gdrive_cfg.get("credentials_file"),
            folder_ids=gdrive_cfg.get("folder_ids", []),
        )

    if connector is None:
        print("Error: no connector enabled in config", file=sys.stderr)
        sys.exit(1)

    chunker = ContextualChunker(
        chunk_size=chunk_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
        context_window=chunk_cfg.get("context_window", 150),
    )
    embedder = Embedder(model=ant_cfg.get("embedding_model", "voyage-3"), batch_size=ant_cfg.get("batch_size", 64))
    retriever = HybridRetriever(embedder=embedder, alpha=retr_cfg.get("alpha", 0.6))

    state_path = Path(args.config).parent / "index_state.json"
    indexer = DocumentIndexer(connector=connector, chunker=chunker, retriever=retriever, state_path=state_path)

    if args.incremental:
        count = indexer.index_incremental(folder_id=args.folder)
        print(f"Incremental index complete: {count} chunks")
    else:
        count = indexer.index_all(folder_id=args.folder)
        print(f"Full index complete: {count} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNTXT — context retrieval daemon", prog="cntxt")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("serve", help="Start the daemon")
    sp.add_argument("--config", default="config/config.yaml")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)

    qp = sub.add_parser("query", help="One-shot query (no daemon)")
    qp.add_argument("--config", default="config/config.yaml")
    qp.add_argument("--prompt", required=True)
    qp.add_argument("--top-k", type=int, default=5, dest="top_k")
    qp.add_argument("--folder", default=None)

    ip = sub.add_parser("index", help="Run index pass")
    ip.add_argument("--config", default="config/config.yaml")
    ip.add_argument("--folder", default=None)
    ip.add_argument("--incremental", action="store_true")

    args = parser.parse_args()

    if args.command == "serve":
        handle_serve(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "index":
        handle_index(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
