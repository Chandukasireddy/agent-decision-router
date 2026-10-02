"""FastMCP Server exposing route_next_action for Claude Code, Cline, Cursor, Codex, and other AI agents."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP

from agent_decision_router.engine import DecisionEngine
from agent_decision_router.models import RouteRequest
from agent_decision_router.registry import create_default_coding_registry

# Instantiate the FastMCP server
mcp = FastMCP(
    "agent-decision-router",
    instructions="A confidence-aware decision layer routing AI coding agent actions using System 1 fast-path and bounded lookahead."
)

# Global router instance for the MCP server process
_engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    """Retrieve or initialize the MCP server's DecisionEngine."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine(registry=create_default_coding_registry())
    return _engine


@mcp.tool(
    name="route_next_action",
    description="Evaluate current agent state, recent tool outputs, and environment to route the optimal next tool action with confidence gating."
)
def route_next_action(
    state: str,
    candidate_tools: Optional[List[str]] = None,
    skip_cache: bool = False,
    confidence_threshold_high: Optional[float] = None,
    confidence_threshold_low: Optional[float] = None,
) -> Dict[str, Any]:
    """Route the next tool action given agent state context.

    Args:
        state: Comprehensive description of the current task, recent tool outputs, git status, and environment errors.
        candidate_tools: Optional subset list of tools to consider. If omitted, all registered tools are considered.
        skip_cache: If True, bypass the Stage 1 fast-path cache and force fresh System 1 evaluation.
        confidence_threshold_high: Optional override for immediate execution threshold (default 0.70).
        confidence_threshold_low: Optional override for declining weak matches (default 0.40).

    Returns:
        Dictionary containing the routing decision, confidence score, status, destructive flag, and rationale.
    """
    engine = get_engine()
    req = RouteRequest(
        state=state,
        candidate_tools=candidate_tools,
        skip_cache=skip_cache,
        confidence_threshold_high=confidence_threshold_high,
        confidence_threshold_low=confidence_threshold_low,
    )
    result = engine.route_sync(req)
    return result.model_dump()


@mcp.tool(
    name="register_agent_tool",
    description="Dynamically register a new tool with custom criteria in the decision router."
)
def register_agent_tool(
    name: str,
    description: str,
    criteria: str,
    is_destructive: bool = False,
    parameters_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a new tool into the routing registry.

    Args:
        name: Unique tool identifier.
        description: Human-readable description.
        criteria: Routing criteria explaining when the decision model should trigger this tool.
        is_destructive: Whether this tool mutates disk, deletes files, or terminates processes.
        parameters_json: Optional JSON string of parameter schemas.

    Returns:
        Status dictionary confirming registration.
    """
    engine = get_engine()
    params = {}
    if parameters_json:
        try:
            params = json.loads(parameters_json)
        except Exception:
            params = {}

    tool = engine.registry.register_tool(
        name=name,
        description=description,
        criteria=criteria,
        parameters=params,
        is_destructive=is_destructive,
    )
    return {
        "status": "registered",
        "tool": tool.model_dump(),
        "total_registered_tools": len(engine.registry.list_tools()),
    }


@mcp.tool(
    name="list_agent_tools",
    description="List all currently registered tools and their routing criteria."
)
def list_agent_tools() -> List[Dict[str, Any]]:
    """List all registered tools."""
    engine = get_engine()
    return [t.model_dump() for t in engine.registry.list_tools()]


@mcp.tool(
    name="get_router_cache_stats",
    description="Get performance metrics and telemetry for the Stage 1 Fast-Path Cache."
)
def get_router_cache_stats() -> Dict[str, Any]:
    """Get fast-path cache statistics."""
    engine = get_engine()
    stats = engine.cache.stats.to_dict()
    stats["current_cache_size"] = engine.cache.size()
    return stats


@mcp.tool(
    name="clear_router_cache",
    description="Clear all entries from the Stage 1 Fast-Path Cache."
)
def clear_router_cache() -> Dict[str, Any]:
    """Clear fast-path cache."""
    engine = get_engine()
    engine.cache.clear()
    return {"status": "cleared", "current_cache_size": engine.cache.size()}


def run_server(transport: str = "stdio") -> None:
    """Run the FastMCP server."""
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")


if __name__ == "__main__":
    run_server()
