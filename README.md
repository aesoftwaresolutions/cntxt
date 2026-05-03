# CNTXT

Transient context retrieval for LLM prompts. Queries iCloud Drive and Google Drive using hybrid semantic + keyword search, then injects the most relevant document chunks into your prompt — nothing stored on disk between runs.

## How it works

```
prompt → query drive APIs → chunk + embed (in-memory) → hybrid rank → inject → model
```

1. A prompt arrives (from an editor plugin or CLI)
2. Connected drives are scanned for documents changed since the last index pass
3. Documents are chunked with Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) method — each chunk gets the document summary prepended before embedding
4. Chunks are ranked by fusing cosine vector similarity (60%) and BM25 keyword scores (40%) via Reciprocal Rank Fusion
5. Top chunks are injected into the prompt as a `<context>` block, then sent to the model

## Setup

```bash
pip install -r requirements.txt
cp config/.env.example .env
# fill in ANTHROPIC_API_KEY
```

Edit `config/config.yaml` to point at your iCloud root or Google Drive folder.

## Usage

```bash
# Start the daemon (FastAPI on localhost:8765)
python main.py serve

# One-shot query
python main.py query --prompt "what are my project deadlines?"

# Force a full re-index
python main.py index

# Incremental re-index (only changed files)
python main.py index --incremental
```

## API (daemon mode)

```
POST /query        { "prompt": str, "top_k": int }  →  { "injected_prompt": str, "sources": [...] }
GET  /health       →  { "status": "ok", "chunks_indexed": int }
POST /reindex      →  { "chunks": int }
```

## Architecture

```
src/
├── connectors/     iCloudConnector, GoogleDriveConnector
├── core/
│   ├── chunker.py  Token-aware split + contextual prefix (tiktoken)
│   ├── embedder.py Anthropic voyage-3 embeddings, batched
│   ├── retriever.py Hybrid RRF: cosine vector + BM25Okapi
│   └── injector.py <context> block builder, token-capped
├── indexing/       DocumentIndexer (full + incremental), ChangeDetector (5-min poll)
└── server/         FastAPI daemon
```

## Configuration

See `config/config.yaml`. Key knobs:

| Key | Default | Effect |
|-----|---------|--------|
| `retrieval.alpha` | `0.6` | Vector weight in RRF fusion (0 = pure BM25, 1 = pure vector) |
| `retrieval.top_k` | `5` | Chunks injected per query |
| `retrieval.max_tokens` | `4000` | Token cap on the context block |
| `chunking.chunk_size` | `512` | Tokens per chunk |
| `chunking.chunk_overlap` | `64` | Overlap between adjacent chunks |
| `chunking.context_window` | `150` | Doc-summary tokens prepended to each chunk |
