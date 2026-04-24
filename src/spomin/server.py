"""MCP server exposing memory tools over Streamable HTTP."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware

from spomin.config import Settings
from spomin.graph import GraphMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spomin")

settings = Settings()
memory = GraphMemory(settings)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Initialize Neo4j indices on startup, cleanup on shutdown."""
    logger.info("Building Neo4j indices and constraints …")
    try:
        await memory.initialize()
        logger.info("Neo4j ready.")
    except Exception:
        logger.warning(
            "Could not connect to Neo4j at %s. "
            "Tools will fail until it becomes available.",
            settings.neo4j_uri,
        )
    yield
    logger.info("Shutting down …")
    try:
        await memory.close()
    except Exception:
        pass


mcp = FastMCP("spomin", lifespan=lifespan, streamable_http_path="/")


@mcp.tool()
async def add_memory(text: str, source: str = "conversation") -> str:
    """Add a memory from natural language text.

    The LLM extracts entities and relationships, which are stored as graph
    nodes and edges in Neo4j.

    Args:
        text: Natural language description of what to remember.
        source: Where this memory came from (e.g. 'conversation', 'document').
    """
    result = await memory.add_episode(text, source)
    return (
        f"Memory stored (uuid={result['episode_uuid']}). "
        f"Extracted {result['node_count']} entities, "
        f"{result['edge_count']} relationships."
    )


@mcp.tool()
async def search_memory(query: str, limit: int = 5) -> str:
    """Search memories semantically.

    Args:
        query: What to search for.
        limit: Maximum number of results to return.
    """
    facts = await memory.search(query, limit)
    if not facts:
        return "No relevant memories found."

    lines = []
    for fact in facts:
        lines.append(
            f"{fact['source']} --[{fact['relation']}]--> {fact['target']}: {fact['fact']}"
        )

    return "\n".join(lines)


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
