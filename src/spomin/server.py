"""MCP server exposing memory tools over Streamable HTTP."""

import json
import logging
from collections.abc import AsyncGenerator
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
async def lifespan(_server: FastMCP) -> AsyncGenerator[None]:
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
    tier: str = "archive",
    project: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """Store raw memory text and queue its chunks for embedding.

    Args:
        text: Natural language description of what to remember.
        tier: Either 'archive' or 'core'. Core is for a small set of durable facts.
        project: Optional project name used to group and filter memories.
        conversation_id: Optional id used to group consecutive chat messages.
    """
    result = await memory.add_memory(
        text,
        "conversation",
        conversation_id,
        project=project,
        tier=tier,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def search_memory(
    query: str,
    limit: int = 5,
    project: str | None = None,
    tier: str | None = None,
) -> str:
    """Search memories using hybrid vector, keyword, and recency ranking.

    Args:
        query: What to search for.
        limit: Maximum number of results to return.
        project: Optional exact project filter.
        tier: Optional 'archive' or 'core' filter.
    """
    results = await memory.search(
        query,
        limit,
        project=project,
        tier=tier,
    )
    if not results:
        return "No relevant memories found."
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def recent_memories(
    limit: int = 10,
    project: str | None = None,
    tier: str | None = None,
) -> str:
    """Return recently stored raw memories, newest first.

    Args:
        limit: Maximum number of memories to return.
        project: Optional exact project filter.
        tier: Optional 'archive' or 'core' filter.
    """
    results = await memory.recent(
        limit,
        project=project,
        tier=tier,
    )
    if not results:
        return "No memories stored."
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def forget_memory(memory_id: str) -> str:
    """Delete a memory by the id returned from search or recent results."""
    deleted = await memory.forget(memory_id)
    if deleted:
        return f"Memory {memory_id} forgotten."
    return f"Memory {memory_id} not found."


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
