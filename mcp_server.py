#!/usr/bin/env python3
"""CNTXT MCP server — exposes retrieve_context as a Claude Code tool."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.connectors.icloud import iCloudConnector
from src.core.chunker import ContextualChunker
from src.core.retriever import HybridRetriever
from src.core.injector import ContextInjector

server = Server("cntxt")

_connector: iCloudConnector | None = None
_chunker: ContextualChunker | None = None
_retriever: HybridRetriever | None = None
_injector: ContextInjector | None = None
_indexed = False


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _init() -> None:
    global _connector, _chunker, _retriever, _injector
    cfg = _load_config()

    icloud_cfg = cfg.get("icloud", {})
    chunk_cfg = cfg.get("chunking", {})
    retr_cfg = cfg.get("retrieval", {})

    _connector = iCloudConnector(
        root=icloud_cfg.get("root"),
        folders=icloud_cfg.get("folders", []),
    )
    _chunker = ContextualChunker(
        chunk_size=chunk_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
        context_window=chunk_cfg.get("context_window", 150),
    )
    # BM25-only — no embedder needed, works without any API key
    _retriever = HybridRetriever(embedder=None, alpha=0.0)
    _injector = ContextInjector(max_tokens=retr_cfg.get("max_tokens", 4000))


def _ensure_indexed() -> None:
    global _indexed
    if _indexed:
        return
    if _retriever is None:
        _init()

    docs = _connector.list_documents()
    all_chunks = []
    errors = []

    for doc in docs:
        try:
            text = _connector.read_document(doc)
            if text.strip():
                all_chunks.extend(_chunker.chunk_document(doc.id, doc.name, text))
        except Exception as e:
            errors.append(f"{doc.name}: {e}")

    _retriever.index(all_chunks)
    _indexed = True

    if errors:
        sys.stderr.write(f"[cntxt] Skipped {len(errors)} files during indexing\n")
    sys.stderr.write(f"[cntxt] Indexed {len(all_chunks)} chunks from {len(docs)} documents\n")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="retrieve_context",
            description=(
                "Search your iCloud documents and return the most relevant excerpts "
                "for a given query. Use this to pull context from notes, docs, code, "
                "or any files synced to iCloud Drive before answering questions about "
                "your projects."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — phrase it like a question or keyword you're looking for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "retrieve_context":
        raise ValueError(f"Unknown tool: {name}")

    _ensure_indexed()

    query = arguments["query"]
    top_k = int(arguments.get("top_k", 5))

    results = _retriever.query(query, top_k=top_k)

    if not results:
        return [types.TextContent(type="text", text="No relevant documents found.")]

    context = _injector.build_context_block(results)
    return [types.TextContent(type="text", text=context)]


async def main() -> None:
    _ensure_indexed()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
