# Spomin

Self-hosted graph memory MCP server for AI agents. Stores knowledge as a graph of entities and relationships, exposed over Streamable HTTP.

"Spomin" is Slovenian for "memory".

## Architecture

```
opencode / llama.cpp webui
           │  Streamable HTTP (port 8084)
           ▼
     spomin server  (FastMCP)
           │
           ▼
         Neo4j        embedding server
```

- **Entity extraction** — LLM via OpenAI-compatible API
- **Semantic search** — embedding server via OpenAI-compatible API
- **Storage** — Neo4j, managed by graphiti-core

## Quick Start

```bash
# 1. Copy and edit config
cp .env.example .env

# 2. Start Neo4j and spomin
docker compose up -d

# 3. Connect your AI client to http://localhost:8084
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
# Neo4j connection
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM for entity extraction (any OpenAI-compatible endpoint)
EXTRACTION_LLM_BASE_URL=http://host.docker.internal:8082/v1
EXTRACTION_LLM_MODEL=Qwen3-0.5B-Q4_K_M.gguf
EXTRACTION_LLM_API_KEY=dummy

# Embedding model for semantic search
EMBEDDER_BASE_URL=http://host.docker.internal:8081/v1
EMBEDDER_MODEL=embeddinggemma-300M-Q8_0.gguf
EMBEDDER_API_KEY=dummy

# Server
HOST=0.0.0.0
PORT=8084
USER_ID=default
```

## MCP Tools

### `add_memory`

Extract entities and relationships from natural language text.

```
add_memory(text, source?) → "Memory stored (uuid=…). Extracted N entities, M relationships."
```

### `search_memory`

Search memories semantically by query.

```
search_memory(query, limit?) → "Entity --[relation]--> Entity: fact"
```

## Requirements

- Docker / Docker Compose
- External LLM server (OpenAI-compatible, e.g. llama-server)
- External embedding server (OpenAI-compatible, e.g. llama-server)

## Dependencies

- `mcp[cli]>=1.0` — MCP SDK
- `graphiti-core>=0.3` — Knowledge graph engine
- `openai>=1.0` — LLM/embedding client
- `pydantic-settings>=2.0` — Config management

## Notes

- Inspect the graph via Neo4j Browser at `http://localhost:7474`
- `host.docker.internal` maps to the host machine for reaching external services
