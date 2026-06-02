# Spomin Memory Plan

## Goal

Spomin should provide shared local memory for AI clients such as opencode and
llama.cpp web UIs without requiring an LLM job for every memory write.

The memory system should be cheap to write to, searchable later, and suitable
for storing full chat history plus durable personal/project facts.

## Core Decision

Use embedding-only episodic memory instead of LLM-extracted graph memory.

The current graph approach asks an LLM to turn each chunk of text into entities
and relationships before storage. That is too slow for local use, unreliable
with small extraction models, and interferes with llama.cpp context caching by
forcing additional generation jobs.

The replacement should store conversation text directly, embed it with a small
embedding model, and retrieve relevant snippets only when needed.

## Storage Model

Store all chats by default.

- Keep every raw message as the source of truth.
- Create searchable chunks from groups of messages rather than embedding every
  tiny message independently.
- Use chunks around 300-800 tokens for normal text, with larger rolling
  conversation windows up to about 1,000-2,000 tokens when useful.
- Attach each chunk back to its original messages, conversation id, timestamp,
  app, model, role, and source.
- Embed chunks asynchronously so chat latency is not blocked by memory writes.

This makes storage append-only and cheap while still allowing exact history to
be recovered later.

## Retrieval Model

Use hybrid retrieval:

- Vector similarity from the local embedding model.
- Keyword search with SQLite FTS5/BM25.
- Recency and access-frequency boosts.
- Optional source filters for client, project, conversation, or time range.

The LLM should not decide what exists in storage. It should only receive the top
few relevant retrieved snippets.

## Memory Tiers

Use two tiers of memory:

1. Archive memory
   - All raw conversations and embedded chunks.
   - Used for broad search and long-term recall.

2. Core memory
   - Small set of durable facts such as user preferences, active projects,
     machine setup, recurring instructions, and important personal context.
   - Can be manually pinned or promoted by a later background process.

Core memory should stay small enough to search often or include directly when
appropriate.

## Preferred Implementation

Replace Neo4j and graphiti-core with a lighter local store:

- SQLite for raw messages, metadata, and FTS5 keyword search.
- sqlite-vec for local vector search, or LanceDB if simpler vector operations
  are preferred.
- Existing OpenAI-compatible embedding server for `/v1/embeddings`.
- FastMCP Streamable HTTP server stays as the integration layer.

Initial MCP tools:

- `add_memory(text, source, conversation_id?)`
- `search_memory(query, limit?)`
- `recent_memories(limit?)`
- `forget_memory(id)`
- `memory_context(query, limit?)`

The first version should avoid extraction LLMs, graph construction, and
cross-encoder reranking. Those can be added later as optional background jobs if
they prove useful.

## Operational Notes

Embedding storage is acceptable for a personal local archive. A 768-dimension
float32 embedding costs about 3 KB per chunk before index overhead, so chunking
by conversation windows keeps the database manageable.

Old chats can be compacted later with background summaries, but summaries should
not replace the raw archive. They should be an additional retrieval aid.
