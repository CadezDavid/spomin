"""MCP server exposing memory tools over Streamable HTTP."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware

from spomin.config import Settings
from spomin.memory import EpisodicMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spomin")

settings = Settings()
memory = EpisodicMemory(settings)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Initialize the local archive and embedding worker."""
    logger.info("Opening memory archive at %s", settings.database_path)
    try:
        await memory.initialize()
        logger.info("Memory archive ready.")
    except Exception:
        logger.exception("Could not initialize the memory archive.")
        raise
    yield
    logger.info("Shutting down.")
    await memory.close()


mcp = FastMCP("spomin", lifespan=lifespan, streamable_http_path="/")


@mcp.tool()
async def add_memory(
    text: str,
    source: str = "conversation",
    conversation_id: str | None = None,
    app: str | None = None,
    model: str | None = None,
    role: str | None = None,
    project: str | None = None,
    tier: str = "archive",
) -> str:
    """Store raw memory text and queue its chunks for embedding.

    Args:
        text: Natural language description of what to remember.
        source: Where this memory came from (e.g. 'conversation', 'document').
        conversation_id: Optional id tying messages to the same conversation.
        app: Optional client application name.
        model: Optional model name associated with the message.
        role: Optional message role such as user or assistant.
        project: Optional project name used for filtering.
        tier: Either 'archive' or 'core'. Core is for a small set of durable facts.
    """
    result = await memory.add_memory(
        text,
        source,
        conversation_id,
        app=app,
        model=model,
        role=role,
        project=project,
        tier=tier,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def search_memory(
    query: str,
    limit: int = 5,
    source: str | None = None,
    app: str | None = None,
    conversation_id: str | None = None,
    project: str | None = None,
    tier: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> str:
    """Search memories using hybrid vector, keyword, and recency ranking.

    Args:
        query: What to search for.
        limit: Maximum number of results to return.
        source: Optional exact source filter.
        app: Optional exact client application filter.
        conversation_id: Optional exact conversation filter.
        project: Optional exact project filter.
        tier: Optional 'archive' or 'core' filter.
        since: Optional inclusive ISO-8601 lower timestamp bound.
        until: Optional inclusive ISO-8601 upper timestamp bound.
    """
    results = await memory.search(
        query,
        limit,
        source=source,
        app=app,
        conversation_id=conversation_id,
        project=project,
        tier=tier,
        since=since,
        until=until,
    )
    if not results:
        return "No relevant memories found."
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def recent_memories(
    limit: int = 10,
    source: str | None = None,
    app: str | None = None,
    conversation_id: str | None = None,
    project: str | None = None,
    tier: str | None = None,
) -> str:
    """Return recently stored raw memories, newest first."""
    results = await memory.recent(
        limit,
        source=source,
        app=app,
        conversation_id=conversation_id,
        project=project,
        tier=tier,
    )
    if not results:
        return "No memories stored."
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def forget_memory(id: str) -> str:
    """Delete a raw memory and its chunks, or delete one chunk by id."""
    deleted = await memory.forget(id)
    return f"Memory {id} forgotten." if deleted else f"Memory {id} not found."


@mcp.tool()
async def memory_context(
    query: str,
    limit: int = 5,
    source: str | None = None,
    app: str | None = None,
    conversation_id: str | None = None,
    project: str | None = None,
    tier: str | None = None,
) -> str:
    """Return compact retrieved snippets suitable for model context."""
    results = await memory.search(
        query,
        limit,
        source=source,
        app=app,
        conversation_id=conversation_id,
        project=project,
        tier=tier,
    )
    if not results:
        return "No relevant memory context."
    blocks = []
    for result in results:
        metadata = [
            f"id={result['id']}",
            f"source={result['source']}",
            f"tier={result['tier']}",
            f"created_at={result['created_at']}",
        ]
        if result["conversation_id"]:
            metadata.append(f"conversation_id={result['conversation_id']}")
        if result["project"]:
            metadata.append(f"project={result['project']}")
        blocks.append(f"[memory {' '.join(metadata)}]\n{result['text']}")
    return "\n\n".join(blocks)


def main() -> None:
    # Get the inner Starlette app and wrap it with CORS
    inner_app: Any = mcp.streamable_http_app()
    app = CORSMiddleware(
        inner_app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
