"""Tests for FastMCP server tools."""

import pytest
from agent_decision_router.mcp_server import (
    get_engine,
    route_next_action,
    register_agent_tool,
    list_agent_tools,
    get_router_cache_stats,
    clear_router_cache,
)
from agent_decision_router.backend.mock import MockBackend


@pytest.fixture(autouse=True)
def setup_mcp_engine():
    engine = get_engine()
    engine.backend = MockBackend(canned_action="run_terminal_command", canned_confidence=0.85)
    engine.cache.clear()
    yield engine


def test_mcp_route_next_action():
    result = route_next_action(
        state="Run pytest on tests/test_auth.py",
        skip_cache=True,
    )
    assert isinstance(result, dict)
    assert result["action"] == "run_terminal_command"
    assert result["confidence"] == 0.85
    assert result["status"] == "EXECUTABLE"


def test_mcp_register_and_list_tools():
    initial_tools = list_agent_tools()
    initial_count = len(initial_tools)

    reg_result = register_agent_tool(
        name="deploy_preview_env",
        description="Deploy preview branch to staging cluster",
        criteria="Deploy preview branch or staging environment",
        is_destructive=True,
    )
    assert reg_result["status"] == "registered"

    tools_after = list_agent_tools()
    assert len(tools_after) == initial_count + 1
    assert any(t["name"] == "deploy_preview_env" for t in tools_after)


def test_mcp_cache_stats_and_clear():
    # Route to populate cache
    _ = route_next_action(state="Inspect config file", skip_cache=False)

    stats = get_router_cache_stats()
    assert "hits" in stats
    assert "current_cache_size" in stats

    clear_res = clear_router_cache()
    assert clear_res["status"] == "cleared"
    assert clear_res["current_cache_size"] == 0
