import os

import pytest

os.environ["DATABASE_PATH"] = "/tmp/spomin-server-tests.db"

pytest.importorskip("mcp")

from spomin.server import mcp


def test_mcp_tool_surface_is_small() -> None:
    tools = mcp._tool_manager._tools

    assert set(tools) == {
        "add_memory",
        "search_memory",
        "recent_memories",
        "forget_memory",
    }
    assert list(tools["add_memory"].parameters["properties"]) == [
        "text",
        "tier",
        "project",
        "conversation_id",
    ]
    assert list(tools["search_memory"].parameters["properties"]) == [
        "query",
        "limit",
        "project",
        "tier",
    ]
    assert list(tools["recent_memories"].parameters["properties"]) == [
        "limit",
        "project",
        "tier",
    ]
    assert list(tools["forget_memory"].parameters["properties"]) == ["memory_id"]
