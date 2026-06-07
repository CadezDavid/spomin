# Spomin

Self-hosted episodic memory MCP server for AI agents. Spomin stores raw text in
SQLite, indexes it with FTS5, embeds chunks through an OpenAI-compatible
endpoint, and retrieves memories with hybrid keyword and vector ranking.

"Spomin" is Slovenian for "memory".

## Architecture

```text
opencode / llama.cpp web UI
           |
           | Streamable HTTP (port 8084)
           v
     Spomin FastMCP server
       |             |
       v             v
 SQLite + FTS5   embedding server
```

- Raw messages are the append-only source of truth.
- Paragraph-aware chunks target 500 tokens and cap at 800 by default.
- Writes commit before embedding and are searchable through FTS immediately.
- A persistent background queue embeds pending chunks in batches.
- Retrieval combines vector and BM25 ranks with recency, access, and core-memory
  boosts.
- No extraction LLM, graph database, or reranker is required.

## Quick Start

Spomin requires an OpenAI-compatible embedding server. For example, start
`llama-server` with an embedding model on port `8081`, then:

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f spomin
```

The Streamable HTTP MCP endpoint is `http://localhost:8084/`.

An OpenAI-compatible embedding endpoint must be reachable at
`EMBEDDER_BASE_URL`. With Docker, `host.docker.internal` resolves to the host.
Keyword search remains available while the embedding endpoint is offline, and
failed embedding jobs are retried.

Stop the server without deleting stored memories:

```bash
docker compose down
```

## Run Locally

Create an environment and install Spomin:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

Change these values in `.env`:

```env
DATABASE_PATH=./data/spomin.db
EMBEDDER_BASE_URL=http://localhost:8081/v1
```

Start the server:

```bash
.venv/bin/spomin
```

## Connect a Client

Configure Spomin as a remote Streamable HTTP MCP server. A typical client
configuration looks like:

```json
{
  "mcpServers": {
    "spomin": {
      "type": "streamable-http",
      "url": "http://localhost:8084/"
    }
  }
}
```

Client field names vary. Use the endpoint URL above when a client asks for a
remote MCP or Streamable HTTP server.

## Configuration

```env
DATABASE_PATH=/app/data/spomin.db

EMBEDDER_BASE_URL=http://host.docker.internal:8081/v1
EMBEDDER_MODEL=embeddinggemma-300M-Q8_0.gguf
EMBEDDER_API_KEY=dummy

CHUNK_TARGET_TOKENS=500
CHUNK_MAX_TOKENS=800
EMBEDDING_BATCH_SIZE=16
EMBEDDING_POLL_INTERVAL=2
EMBEDDING_RETRY_INTERVAL=15

HOST=0.0.0.0
PORT=8084
```

For local execution outside Docker, set `DATABASE_PATH=./data/spomin.db` and
point `EMBEDDER_BASE_URL` at a host-reachable URL.

Additional retrieval weights such as `VECTOR_WEIGHT`, `KEYWORD_WEIGHT`,
`RECENCY_WEIGHT`, and `CORE_MEMORY_BOOST` can be set through environment
variables. Defaults are defined in `src/spomin/config.py`.

## MCP Tools

### `add_memory`

Stores raw text and queues chunks for embedding.

```text
add_memory(
  text,
  tier="archive",
  project?,
  conversation_id?
)
```

Use `tier="core"` for a small set of durable preferences, project facts,
machine details, or recurring instructions.

`project` groups memories that belong to the same project. `conversation_id`
lets Spomin combine consecutive messages from one chat into rolling chunks.
Both are optional.

Example:

```text
add_memory(
  text="I prefer Python for backend services.",
  tier="core",
  project="spomin"
)
```

### `search_memory`

Returns structured JSON results from hybrid retrieval.

```text
search_memory(
  query,
  limit=5,
  project?,
  tier?
)
```

`project` and `tier` are optional exact filters.

Example:

```text
search_memory(
  query="Which language do I prefer for backend work?",
  limit=5
)
```

### `recent_memories`

Returns recently stored raw messages, newest first. It accepts optional
`limit`, `project`, and `tier` arguments.

### `forget_memory`

Deletes a memory using an id returned by `search_memory` or `recent_memories`.

```text
forget_memory(memory_id="9f784078-c5d4-4806-a26a-8f38949aea95")
```

## Development

```bash
.venv/bin/pytest
python -m compileall -q src tests
docker compose config --quiet
```

SQLite must include FTS5 support. Current official Python and Docker images do.

## Storage

The `spomin_data` Docker volume contains the SQLite database. Embeddings are
stored as float32 blobs alongside their dimensions. Raw messages are never
replaced by summaries; later compaction or summarization can be added as
separate retrieval aids.

Deleting the Docker volume permanently removes the archive:

```bash
docker compose down --volumes
```

Vector search currently scans stored embeddings in-process. This keeps the
implementation simple and is appropriate for a personal archive. A vector
index such as sqlite-vec can be added if archive size makes scan latency
measurable.

Potential future additions include optional background summaries, core-memory
promotion workflows, and archive export/import commands. Summaries should
remain retrieval aids and never replace the raw message archive.
