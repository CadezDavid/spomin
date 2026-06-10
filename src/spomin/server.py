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
    """Store important facts, user preferences, or key takeaways to memory for long-term recall.
    Use this proactively whenever the user shares something significant or you derive a useful insight that should be remembered across sessions.

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
    """Retrieve relevant information from long-term memory.
    Call this whenever the user mentions past events, preferences, or topics that are likely stored in memory, or when you need context from previous conversations.

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
    """List the most recently stored memories.
    Useful for reviewing the last few things learned or ensuring recent updates were saved correctly.

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
    """Remove a specific memory from storage.
    Use this when the user explicitly asks to forget something or when information is corrected/outdated.
    """
    deleted = await memory.forget(memory_id)
    if deleted:
        return f"Memory {memory_id} forgotten."
    return f"Memory {memory_id} not found."


@mcp.prompt()
def memory_usage() -> str:
    """Instructions for the AI on how to manage long-term memory effectively."""
    return (
        "You are equipped with a long-term memory system (spomin). To provide a personalized "
        "and consistent experience, you should manage this memory proactively:\n\n"
        "1. **Saving Memories (`add_memory`)**:\n"
        "   - Whenever the user shares a preference, a personal detail, a goal, or a specific "
        "way they like things done, save it.\n"
        "   - After completing a complex task or reaching a milestone, save a summary of the "
        "outcome and key learnings.\n"
        "   - If you discover a fact about the user or the project that isn't common knowledge, store it.\n"
        "   - Use 'core' tier for fundamental truths (e.g., 'The user prefers Python over JS') "
        "and 'archive' for episodic details.\n\n"
        "2. **Recalling Memories (`search_memory`, `recent_memories`)**:\n"
        "   - At the start of a new topic or session, search for relevant context.\n"
        "   - When the user says 'remember when...', 'as I mentioned before...', or 'my usual...', "
        "search memory immediately.\n"
        "   - If you are unsure about a preference, check memory before asking the user.\n\n"
        "3. **Maintaining Memories (`forget_memory`)**:\n"
        "   - If the user corrects a previous statement, update the memory by searching for the "
        "old one and forgetting it, then adding the new one.\n\n"
        "Your goal is to make the user feel known and understood without them having to repeat themselves."
    )


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
